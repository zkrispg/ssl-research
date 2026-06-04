"""One-off: merge STARSS22 cross-eval per_ckpt dicts from the v4 handoff
backup (MIC+CRNN cells 110/111/112/113) with the current file
(MIC+Xfm cells 140/141/142), then re-aggregate and rewrite JSON + MD.

A prior `--cells 140 141 142` rerun overwrote the JSON, dropping the
MIC+CRNN cross-eval results. This restores them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
import _path_c_cross_starss22 as cross

PAPER   = Path(r"D:\ssl-research\paper")
CURRENT = PAPER / "path_c_cross_starss22.json"
BACKUP  = PAPER / "_handoff_v4_20260526_090338" / "path_c_cross_starss22.json"


def main() -> int:
    cur = json.loads(CURRENT.read_text(encoding="utf-8"))
    bak = json.loads(BACKUP.read_text(encoding="utf-8"))

    merged_per_ckpt = {}
    # Backup first (MIC+CRNN), then current (MIC+Xfm) so current wins on conflict.
    merged_per_ckpt.update(bak.get("per_ckpt", {}))
    merged_per_ckpt.update(cur.get("per_ckpt", {}))
    print(f"[merge] backup ckpts={len(bak.get('per_ckpt', {}))}, "
          f"current ckpts={len(cur.get('per_ckpt', {}))}, "
          f"merged={len(merged_per_ckpt)}")

    payload = cross.aggregate(merged_per_ckpt)
    payload["per_ckpt"] = merged_per_ckpt

    CURRENT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    cross.write_md_summary(payload, PAPER / "path_c_cross_starss22.md")
    print(f"[saved] {CURRENT}")
    print(f"[saved] {PAPER / 'path_c_cross_starss22.md'}")

    # Sanity: print which cells now have data
    print("\n[verify] per-cell n_seeds:")
    for cname, cell in payload["per_cell"].items():
        print(f"  {cname:<26} n={cell['n_seeds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
