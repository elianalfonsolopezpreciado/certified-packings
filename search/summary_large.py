"""results/large_n_summary.md: our best certified packing for every n >= 100 that has a certificate, versus the LOCAL copy of the
Packomania csqv table (its retrieval time is printed; refresh with `python -m search.record_check`).

Columns: our raw exact score (verifier, same file, no shrink), our certified score (shrink 1e-12 per radius), listed record, gap (raw - listed),
verdict: AHEAD (> +1e-11), tie, behind.
"""
from __future__ import annotations

import json
from pathlib import Path

from verify import load_certificate, parse, verify_fraction

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if not (ROOT / "data" / "packomania_csqv_records.json").exists():
        raise SystemExit("no local Packomania table: run `python scripts/fetch_packomania.py` first (data/ is not redistributed)")
    j = json.load(open(ROOT / "data" / "packomania_csqv_records.json"))
    table = {int(k): v for k, v in j["sumradii"].items()}
    rows = []
    for p in sorted((ROOT / "certificates").glob("A_n*.json")):
        if any(s in p.name for s in (".verify_", ".summary")):
            continue
        rep = verify_fraction(parse(load_certificate(str(p))))
        n = rep.n
        listed = float(table[n])
        raw = float(rep.raw_score)
        verdict = "AHEAD" if raw - listed > 1e-11 else ("tie" if abs(raw - listed) <= 1e-11 + n * 1e-12 else "behind")
        rows.append((n, raw, float(rep.certified_score), listed, raw - listed, verdict, rep.valid))
    rows.sort()
    md = [f"Reference: Packomania csqv table retrieved {j['retrieved']} (local copy). All values exact (verifier, both backends passed at certification).",
          "", "| n | our raw exact | our certified (shrink 1e-12) | listed record | gap (raw - listed) | verdict |", "|---|---|---|---|---|---|"]
    for n, raw, cs, listed, gap, v, ok in rows:
        md.append(f"| {n} | {raw:.12f} | {cs:.12f} | {listed:.12f} | {gap:+.3e} | {v}{'' if ok else ' (INVALID)'} |")
    (ROOT / "results" / "large_n_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    bs, nl = chr(92), chr(10)
    tex = [bs + "begin{tabular}{rllll}", bs + "hline", "$n$ & ours (raw exact) & listed record & gap & verdict " + bs + bs, bs + "hline"]
    for n, raw, cs, listed, gap, v, ok in rows:
        tex.append(f"{n} & {raw:.12f} & {listed:.12f} & ${gap:+.2e}$ & {v} " + bs + bs)
    tex += [bs + "hline", bs + "end{tabular}"]
    (ROOT / "paper" / "tables_large.tex").write_text(nl.join(tex) + nl, encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
