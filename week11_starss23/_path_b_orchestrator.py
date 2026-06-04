"""Watcher / orchestrator for the Path B chain.

This script polls the filesystem and does three things automatically:

    1. When ``foa_dev.zip`` reaches its full Zenodo size (~3.4 GB), it
       is extracted to ``data/STARSS23/foa_dev/`` if not already.
    2. When ``STARSS22/mic_dev.zip`` reaches its full size (~2.1 GB), it
       is extracted to ``data/STARSS22/mic_dev/`` if not already.
    3. Once BOTH preconditions below hold, it launches the v3 supervisor
       and then exits.

Preconditions for the v3 launch:

    (a) G5 (capacity sweep N=5 extension) supervisor has finished, i.e.
        runs/multiseed_summary_capacity_sweep_n5_extension.json exists
        and reports status="complete".
    (b) STARSS23 FOA dev set has been extracted into
        ``D:/ssl-research/data/STARSS23/foa_dev/`` (look for >=80 wav
        files; the dataset has 168 dev clips so this is conservative).

Sleep granularity is 60 seconds; the script is cheap to leave running
detached for many hours. Designed to be safe to relaunch (idempotent
extract + skip-already-extracted).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO = Path("D:/ssl-research")
PYEXE = REPO / "venv" / "Scripts" / "python.exe"
RUNS = REPO / "week11_starss23" / "runs"
LOG = RUNS / "path_b_orchestrator.log"

G5_SUMMARY = RUNS / "multiseed_summary_capacity_sweep_n5_extension.json"
FOA_DIR = REPO / "data" / "STARSS23" / "foa_dev"

FOA_ZIP = REPO / "data" / "STARSS23" / "foa_dev.zip"
FOA_ZIP_TARGET_BYTES = 3406_000_000  # ~3406 MB on Zenodo
STARSS22_MIC_ZIP = REPO / "data" / "STARSS22" / "mic_dev.zip"
STARSS22_MIC_TARGET_BYTES = 2120_000_000  # ~2120 MB on Zenodo
STARSS22_MIC_DIR = REPO / "data" / "STARSS22" / "mic_dev"

POLL_S = 60
MAX_WAIT_HOURS = 48


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def g5_complete() -> bool:
    if not G5_SUMMARY.exists():
        return False
    try:
        data = json.loads(G5_SUMMARY.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("status") == "complete"


def foa_extracted() -> bool:
    if not FOA_DIR.is_dir():
        return False
    n_wavs = sum(1 for _ in FOA_DIR.rglob("*.wav"))
    return n_wavs >= 80


def starss22_mic_extracted() -> bool:
    if not STARSS22_MIC_DIR.is_dir():
        return False
    return sum(1 for _ in STARSS22_MIC_DIR.rglob("*.wav")) >= 100


def _zip_complete(zp: Path, target_bytes: int) -> bool:
    if not zp.exists():
        return False
    return zp.stat().st_size >= int(target_bytes * 0.99)


def _try_extract(zp: Path, dest_root: Path, label: str) -> bool:
    """Attempt to extract ``zp`` to ``dest_root.parent`` (the zip already
    has a top-level directory). Return True on success, False on failure
    (e.g. zip not yet finished downloading)."""
    try:
        with zipfile.ZipFile(zp) as z:
            t0 = time.time()
            z.extractall(dest_root.parent)
            n = len(z.namelist())
        _log(f"[extract] {label}: extracted {n} entries in {time.time()-t0:.1f}s")
        return True
    except zipfile.BadZipFile:
        _log(f"[extract] {label}: zip not yet valid (still downloading?)")
        return False
    except OSError as exc:
        _log(f"[extract] {label}: OSError -- {exc}")
        return False


def maybe_extract_assets() -> None:
    """Extract foa_dev.zip and STARSS22/mic_dev.zip when they are fully
    downloaded and not yet extracted."""
    if not foa_extracted() and _zip_complete(FOA_ZIP, FOA_ZIP_TARGET_BYTES):
        _log(f"[extract] foa_dev.zip is complete; extracting...")
        _try_extract(FOA_ZIP, FOA_DIR, "STARSS23/foa_dev")
    if not starss22_mic_extracted() and _zip_complete(
        STARSS22_MIC_ZIP, STARSS22_MIC_TARGET_BYTES
    ):
        _log(f"[extract] STARSS22/mic_dev.zip is complete; extracting...")
        _try_extract(STARSS22_MIC_ZIP, STARSS22_MIC_DIR, "STARSS22/mic_dev")


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    _log("=" * 72)
    _log(
        f"Path B orchestrator started. Will launch supervisor v3 once both:"
        f"\n  (a) {G5_SUMMARY} status=complete"
        f"\n  (b) {FOA_DIR}/  has >=80 wav files"
    )
    deadline = time.time() + MAX_WAIT_HOURS * 3600.0
    last_status = ""
    while time.time() < deadline:
        # Side task: auto-extract any newly-arrived dataset zips.
        maybe_extract_assets()
        g5_ok = g5_complete()
        foa_ok = foa_extracted()
        s22_ok = starss22_mic_extracted()
        status = f"g5={g5_ok}  foa={foa_ok}  starss22={s22_ok}"
        if status != last_status:
            _log(status)
            last_status = status
        if g5_ok and foa_ok:
            _log("Preconditions satisfied (G5 done + FOA extracted); launching supervisor v3.")
            cmd = [str(PYEXE), "-u", "-m", "week11_starss23._supervisor_chain_v3"]
            rc = subprocess.run(cmd, cwd=str(REPO), check=False).returncode
            _log(f"supervisor v3 returncode={rc}")
            return
        time.sleep(POLL_S)
    _log(f"timed out after {MAX_WAIT_HOURS} hours waiting for preconditions")
    sys.exit(2)


if __name__ == "__main__":
    main()
