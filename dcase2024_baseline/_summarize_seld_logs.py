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


def _parse_csv(value: str, cast=str):
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def _parse_pairs(value: str):
    pairs = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        label, tasks = part.split(":", 1)
        full, no_geom = tasks.split("-", 1)
        pairs.append((label.strip(), full.strip(), no_geom.strip()))
    return pairs


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


def _fmt(value, digits=3, signed=False):
    if value is None:
        return "NA"
    if signed:
        return f"{value:+.{digits}f}"
    return f"{value:.{digits}f}"


def _t_pdf(x: float, nu: int) -> float:
    return (
        math.gamma((nu + 1) / 2)
        / (math.sqrt(nu * math.pi) * math.gamma(nu / 2))
        * (1 + x * x / nu) ** (-(nu + 1) / 2)
    )


def _simpson(f, a: float, b: float, n: int = 20000) -> float:
    if n % 2:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * f(a + i * h)
    return total * h / 3


def _t_two_sided_p(t_value: float, df: int):
    if df <= 0 or math.isnan(t_value):
        return None
    t_value = abs(t_value)
    area = _simpson(lambda x: _t_pdf(x, df), 0.0, t_value)
    return max(0.0, min(1.0, 2 * (1 - (0.5 + area))))


def paired_stats(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return {"n": len(values), "mean": _mean(values), "sd": None, "t": None, "p": None, "dz": None}
    mu = _mean(values)
    sd = _sd(values)
    if not sd:
        return {"n": len(values), "mean": mu, "sd": sd, "t": None, "p": None, "dz": None}
    t_value = mu / (sd / math.sqrt(len(values)))
    return {
        "n": len(values),
        "mean": mu,
        "sd": sd,
        "t": t_value,
        "p": _t_two_sided_p(t_value, len(values) - 1),
        "dz": mu / sd,
    }


def parse_cell(run_dir: Path, task: str, job: str, seed: int):
    train_log = run_dir / f"dcase2024_{task}_{job}.log"
    test_log = run_dir / f"dcase2024_{task}_{job}_test.log"
    row = {
        "task": task,
        "job": job,
        "seed": seed,
        "status": "missing",
        "train_log": str(train_log),
        "test_log": str(test_log),
        "best_epoch": None,
        "best_metric": None,
        "data_digest": None,
        "lr": None,
        "dropout": None,
    }

    if train_log.exists():
        train_text = train_log.read_text(encoding="utf-8", errors="replace")
        row["best_epoch"] = _last_int_match(r"best_val_epoch:\s*([0-9]+)", train_text)
        row["best_metric"] = _last_str_match(r"\[checkpoint\] best_metric = ([a-z]+)", train_text)
        row["data_digest"] = _last_str_match(r"\[data\] test files=\d+ digest=([0-9a-f]+)", train_text)
        row["lr"] = _last_str_match(r"\[run\] SSL_LR applied: lr=([0-9.eE+-]+)", train_text)
        row["dropout"] = _last_str_match(r"\[run\] SSL_DROPOUT applied: dropout_rate=([0-9.eE+-]+)", train_text)

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
    summaries = []
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task and row["status"] == "complete"]
        item = {"task": task, "n": len(task_rows), "seeds": [row["seed"] for row in task_rows]}
        for metric in ("seld", "f20", "doae", "dist", "rde"):
            values = [row.get(metric) for row in task_rows]
            item[f"{metric}_mean"] = _mean(values)
            item[f"{metric}_sd"] = _sd(values)
        summaries.append(item)
    return summaries


def paired_deltas(rows, pairs):
    out = []
    for label, full_task, no_geom_task in pairs:
        full_by_seed = {
            row["seed"]: row for row in rows
            if row["task"] == full_task and row["status"] == "complete"
        }
        no_geom_by_seed = {
            row["seed"]: row for row in rows
            if row["task"] == no_geom_task and row["status"] == "complete"
        }
        for seed in sorted(set(full_by_seed) & set(no_geom_by_seed)):
            item = {"label": label, "pair": f"{full_task}-{no_geom_task}", "seed": seed}
            for metric in ("seld", "f20", "doae", "dist", "rde"):
                a = full_by_seed[seed].get(metric)
                b = no_geom_by_seed[seed].get(metric)
                item[f"delta_{metric}"] = None if a is None or b is None else a - b
            out.append(item)
    return out


def write_markdown(path: Path, title: str, summaries, deltas, rows):
    lines = [
        f"# {title}",
        "",
        "Positive delta means full-geometry task minus matched no-geometry control.",
        "For DOAE/SELD/RDE lower is better; for F20 higher is better.",
        "",
        "## Per Task",
        "",
        "| task | n | seeds | SELD mean | F20 mean | DOAE mean | RDE mean |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['task']} | {item['n']} | {item['seeds']} | "
            f"{_fmt(item.get('seld_mean'))} +/- {_fmt(item.get('seld_sd'))} | "
            f"{_fmt(item.get('f20_mean'))} +/- {_fmt(item.get('f20_sd'))} | "
            f"{_fmt(item.get('doae_mean'))} +/- {_fmt(item.get('doae_sd'))} | "
            f"{_fmt(item.get('rde_mean'))} +/- {_fmt(item.get('rde_sd'))} |"
        )

    lines += [
        "",
        "## Paired Contrasts",
        "",
        "| label | pair | n | delta SELD | delta F20 | delta DOAE | delta RDE | t(DOAE) | p(DOAE) | dz(DOAE) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in sorted({row["label"] for row in deltas}):
        pair_rows = [row for row in deltas if row["label"] == label]
        pair = pair_rows[0]["pair"] if pair_rows else "NA"
        doae_stats = paired_stats([row.get("delta_doae") for row in pair_rows])
        lines.append(
            f"| {label} | {pair} | {len(pair_rows)} | "
            f"{_fmt(_mean([row.get('delta_seld') for row in pair_rows]), signed=True)} | "
            f"{_fmt(_mean([row.get('delta_f20') for row in pair_rows]), signed=True)} | "
            f"{_fmt(_mean([row.get('delta_doae') for row in pair_rows]), signed=True)} | "
            f"{_fmt(_mean([row.get('delta_rde') for row in pair_rows]), signed=True)} | "
            f"{_fmt(doae_stats.get('t'), signed=True)} | "
            f"{_fmt(doae_stats.get('p'))} | "
            f"{_fmt(doae_stats.get('dz'), signed=True)} |"
        )

    lines += [
        "",
        "## Cells",
        "",
        "| task | seed | job | status | best epoch | best metric | SELD | F20 | DOAE | RDE | digest |",
        "|---|---:|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['seed']} | {row['job']} | {row['status']} | "
            f"{row.get('best_epoch') if row.get('best_epoch') is not None else 'NA'} | "
            f"{row.get('best_metric') or 'NA'} | {_fmt(row.get('seld'))} | "
            f"{_fmt(row.get('f20'))} | {_fmt(row.get('doae'))} | {_fmt(row.get('rde'))} | "
            f"{row.get('data_digest') or 'NA'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=repo_root / "runs")
    parser.add_argument("--out-dir", type=Path, default=repo_root / "runs")
    parser.add_argument("--tasks", required=True, help="Comma-separated task ids.")
    parser.add_argument("--seeds", required=True, help="Comma-separated seeds.")
    parser.add_argument("--job-template", default="det_gca_seed{seed}")
    parser.add_argument("--pairs", default="", help="Comma list: label:full-no_geom")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--title", default="SELD Run Summary")
    args = parser.parse_args()

    tasks = _parse_csv(args.tasks, str)
    seeds = _parse_csv(args.seeds, int)
    pairs = _parse_pairs(args.pairs) if args.pairs else []

    rows = []
    for seed in seeds:
        for task in tasks:
            rows.append(parse_cell(args.run_dir, task, args.job_template.format(seed=seed, task=task), seed))

    summaries = summarize_tasks(rows)
    deltas = paired_deltas(rows, pairs)
    payload = {"rows": rows, "summaries": summaries, "deltas": deltas, "pairs": pairs}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{args.out_prefix}.json"
    csv_path = args.out_dir / f"{args.out_prefix}.csv"
    md_path = args.out_dir / f"{args.out_prefix}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(md_path, args.title, summaries, deltas, rows)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
