"""Re-download the live Packomania csqv table and compare it with every certified sum-of-radii packing.

    python -m search.record_check            # downloads, refreshes data/packomania_csqv_records.json (data/ is git-ignored; Packomania data is not redistributed), appends to results/record_checks.csv
    python -m search.record_check --offline  # compare with the current local table only

The table moves fast (hundreds of rows changed within hours on 2026-09-29), so every claim must carry the time of the last check.
The comparison uses the verifier's *raw* exact score (same file, no shrink) and the certified (shrunk by 1e-12) score.
"""
from __future__ import annotations

import csv
import datetime
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.packomania.com/csqv/txt/sumradii.txt"


def parse_table(text: str) -> dict[int, str]:
    d: dict[int, str] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        a = line.split()
        if len(a) >= 2:
            d[int(a[0])] = a[1]
    return d


def main() -> None:
    offline = "--offline" in sys.argv
    now = datetime.datetime.now()
    stamp = now.strftime("%Y-%m-%d %H:%M")
    (ROOT / "data").mkdir(exist_ok=True)
    if not offline:
        with urllib.request.urlopen(URL, timeout=60) as r:
            text = r.read().decode()
        (ROOT / "data" / f"packomania_csqv_sumradii_{now.strftime('%Y-%m-%d_%H%M')}.txt").write_text(text)
        (ROOT / "data" / "packomania_csqv_sumradii.txt").write_text(text)
        table = parse_table(text)
        json.dump({"source": URL, "retrieved": stamp, "sumradii": {str(k): v for k, v in table.items()}},
                  open(ROOT / "data" / "packomania_csqv_records.json", "w"), indent=0)
    else:
        if not (ROOT / "data" / "packomania_csqv_records.json").exists():
            raise SystemExit("no local Packomania table: run `python scripts/fetch_packomania.py` first (data/ is not redistributed)")
        j = json.load(open(ROOT / "data" / "packomania_csqv_records.json"))
        table = {int(k): v for k, v in j["sumradii"].items()}
        stamp = j["retrieved"] + " (local copy)"
    from verify import load_certificate, parse, verify_fraction

    out = ROOT / "results" / "record_checks.csv"
    new = not out.exists()
    rows = []
    for p in sorted((ROOT / "certificates").glob("*.json")):
        if any(s in p.name for s in (".verify_", ".summary", ".eps")):
            continue
        cert = load_certificate(str(p))
        if cert.get("problem") != "sum_radii":
            continue
        rep = verify_fraction(parse(cert))
        n = rep.n
        if n not in table or n < 26:
            continue
        listed = float(table[n])
        raw, cert_s = float(rep.raw_score), float(rep.certified_score)
        rows.append([stamp, p.name, n, f"{raw:.12f}", f"{cert_s:.12f}", table[n], f"{raw - listed:+.3e}",
                     "AHEAD" if raw - listed > 1e-11 else ("tie" if abs(raw - listed) <= 1e-11 + n * 1e-12 else "behind")])
    with open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["checked_at", "certificate", "n", "our_raw", "our_certified", "listed_record", "gap_raw", "verdict"])
        w.writerows(rows)
    for r in rows:
        if r[2] >= 100 or r[7] == "AHEAD":
            print(*r, sep=" | ")


if __name__ == "__main__":
    main()
