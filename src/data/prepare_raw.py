"""Joins and extracts the raw Spark event logs published with the paper.

Source: the split zip "all zip.zip.001" to ".007" in resources/ of github.com/mgarralda/garralda-performance-model.
Writes data/raw/eventlogs.zip, data/raw/eventlogs/<size>/<setting>/<application> and data/raw/MANIFEST.json
with the checksums. Files already extracted are skipped and --budget S stops after S seconds, so it can
be run in pieces.
"""
import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.paths import GARRALDA_REPO, RAW, LOGS_DIR  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1e9, help="stop after this many seconds (resumable)")
    a = ap.parse_args()
    t0 = time.time()
    parts = sorted((GARRALDA_REPO / "resources").glob("all zip.zip.0*"))
    if not parts:
        sys.exit("split zip parts not found under external/garralda-performance-model/resources")
    out = RAW / "eventlogs.zip"
    expected = sum(p.stat().st_size for p in parts)
    if not out.exists() or out.stat().st_size != expected:
        with open(out, "wb") as w:
            for p in parts:
                w.write(p.read_bytes())
    z = zipfile.ZipFile(out)
    files = [i for i in z.infolist() if not i.is_dir()]
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for i in files:
        target = LOGS_DIR / i.filename
        if target.exists() and target.stat().st_size == i.file_size:
            skipped += 1
            continue
        if time.time() - t0 > a.budget:
            break
        z.extract(i, LOGS_DIR)
        done += 1
    remaining = sum(1 for i in files if not ((LOGS_DIR / i.filename).exists()
                                             and (LOGS_DIR / i.filename).stat().st_size == i.file_size))
    status = {"extracted_now": done, "already_present": skipped, "remaining": remaining,
              "seconds": round(time.time() - t0, 1)}
    if remaining == 0:
        bad = z.testzip()
        manifest = {
            "source": "https://github.com/mgarralda/garralda-performance-model (resources/all zip.zip.00*)",
            "parts": [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in parts],
            "zip": {"bytes": out.stat().st_size, "sha256": sha256(out), "crc_test": "ok" if bad is None else f"bad: {bad}"},
            "entries_total": len(z.infolist()), "files": len(files),
            "uncompressed_bytes": sum(i.file_size for i in files),
        }
        (RAW / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
        status["zip_sha256"] = manifest["zip"]["sha256"]
        status["crc_test"] = manifest["zip"]["crc_test"]
    print(json.dumps(status))


if __name__ == "__main__":
    main()
