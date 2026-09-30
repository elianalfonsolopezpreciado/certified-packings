"""CLI:  python -m verify CERT.json [--backend fraction|mpmath|both] [--eps 1e-12] [--out report.json]

Exit code 0 iff every requested backend certifies the file as valid.
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction

from .core import BACKENDS, CertificateError, load_certificate, parse


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m verify")
    ap.add_argument("certificate")
    ap.add_argument("--backend", choices=["fraction", "mpmath", "both"], default="both")
    ap.add_argument("--eps", default="1e-12", help="radius shrink applied before certifying sum_radii")
    ap.add_argument("--out", help="write the JSON report here")
    a = ap.parse_args(argv)
    try:
        parsed = parse(load_certificate(a.certificate))
    except (CertificateError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}))
        return 2
    eps = Fraction(a.eps)
    names = list(BACKENDS) if a.backend == "both" else [a.backend]
    reports = [BACKENDS[b](parsed, eps).to_dict() for b in names]
    for r in reports:
        print(f"[{r['backend']}] problem={r['problem']} n={r['n']} valid={r['valid']} "
              f"certified_score={r['certified_score']} raw_score={r['raw_score']} "
              f"max_violation={r['max_violation']}")
        for m in r["messages"]:
            print("   !", m)
    ok = all(r["valid"] for r in reports)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"certificate": a.certificate, "all_valid": ok, "reports": reports}, fh, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
