"""Download the Packomania tables / coordinate files that our comparisons use into ./data (git-ignored, NOT redistributed).

    python scripts/fetch_packomania.py                 # tables (csq distance/radius, csqv sum of radii) + coordinates of n = 27, 120, 250
    python scripts/fetch_packomania.py --n 120 250     # choose the coordinate files
    python scripts/fetch_packomania.py --tables-only

It writes the raw files plus the two small JSON tables that `search/driver.py` (load_record) reads:
    data/packomania_csq_records.json   {"source","retrieved","distance":{n: "..."},"radius":{n: "..."}}
    data/packomania_csqv_records.json  {"source","retrieved","sumradii":{n: "..."}}
and prints the SHA-256 of every file next to the checksum of the snapshot we used (docs/packomania_snapshot_checksums.json).
Packomania's tables change often (csqv for n >= 100 several times a day), so a MISMATCH is normal and simply means the table moved after our snapshot;
comparisons are only meaningful for the time of the download. Packomania data remains the property of its owner (E. Specht, https://www.packomania.com).
If the site is unreachable the script exits with status 1 and the rest of the repository still works (verification never needs these files).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UA = {"User-Agent": "certified-packings-fetch/1.0 (research comparison; contact via repository)"}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def parse_table(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        a = line.split()
        if len(a) >= 2:
            out[str(int(a[0]))] = a[1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="*", default=[27, 120, 250], help="csqv coordinate files to download")
    ap.add_argument("--tables-only", action="store_true")
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    (DATA / "packomania_files").mkdir(exist_ok=True)
    stamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %z")
    snap = {f["file"]: f["sha256"] for f in json.load(open(ROOT / "docs" / "packomania_snapshot_checksums.json"))["files"]}
    got: dict[str, bytes] = {}
    jobs = {
        "packomania_csq_distance.txt": "http://www.packomania.com/csq/txt/distance.txt",
        "packomania_csq_radius.txt": "http://www.packomania.com/csq/txt/radius.txt",
        "packomania_csqv_sumradii.txt": "https://www.packomania.com/csqv/txt/sumradii.txt",
    }
    if not a.tables_only:
        for n in a.n:
            jobs[f"packomania_files/csqv{n}.txt"] = f"https://www.packomania.com/csqv/txt/csqv{n}.txt"
    try:
        for rel, url in jobs.items():
            got[rel] = get(url)
            (DATA / rel).write_bytes(got[rel])
    except Exception as exc:  # network problems: report, do not crash the caller (Makefile uses '-')
        print(f"could not download from Packomania: {exc}", file=sys.stderr)
        return 1
    json.dump({"source": jobs["packomania_csq_distance.txt"], "retrieved": stamp, "distance": parse_table(got["packomania_csq_distance.txt"].decode()),
               "radius": parse_table(got["packomania_csq_radius.txt"].decode())}, open(DATA / "packomania_csq_records.json", "w"), indent=0)
    json.dump({"source": jobs["packomania_csqv_sumradii.txt"], "retrieved": stamp, "sumradii": parse_table(got["packomania_csqv_sumradii.txt"].decode())},
              open(DATA / "packomania_csqv_records.json", "w"), indent=0)
    ref = {"packomania_csq_distance.txt": "packomania_csq_distance.txt", "packomania_csq_radius.txt": "packomania_csq_radius.txt"}
    print(f"downloaded {len(got)} files at {stamp}")
    for rel, b in got.items():
        h = hashlib.sha256(b).hexdigest()
        key = ref.get(rel, rel if rel in snap else None)
        note = "no snapshot hash recorded for this file (csqv sum-of-radii snapshots are keyed by time)" if key is None else \
               ("identical to our snapshot" if snap[key] == h else "differs from our snapshot (the table moved; expected)")
        print(f"  {rel}: {len(b)} bytes sha256={h[:16]}...  {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
