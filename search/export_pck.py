"""Export a sum_radii certificate to Packomania's `.pck` layout (csqv, unequal circles).

Layout, per http://www.packomania.com/hints.html (read 2026-09-28): line 1 = radius of the LARGEST circle,
line 2 = author(s), then one `x y r` line per circle, sorted by increasing radius, coordinates centred at the
container centre. Verify the layout against the current hints page before sending anything.

    python -m search.export_pck certificates/sum_radii_n27.json "Author Name" --shrink 1e-12 > csqv27.pck

`--shrink` reproduces the verifier's radius shrink (use the same eps you certified with).
Nothing is ever sent by this project.
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("certificate")
    ap.add_argument("author")
    ap.add_argument("--shrink", default="1e-12")
    ap.add_argument("--digits", type=int, default=20)
    a = ap.parse_args()
    cert = json.load(open(a.certificate), parse_float=str)
    eps = Fraction(a.shrink)
    rows = []
    for x, y, r in cert["circles"]:
        rr = max(Fraction(r) - eps, Fraction(0))
        rows.append((Fraction(x) - Fraction(1, 2), Fraction(y) - Fraction(1, 2), rr))
    rows.sort(key=lambda t: t[2])

    from decimal import Decimal, getcontext

    getcontext().prec = 60

    def f(v: Fraction) -> str:
        q = Decimal(v.numerator) / Decimal(v.denominator)
        return format(q, f".{a.digits}f")

    print(f(rows[-1][2]))
    print(a.author)
    for x, y, r in rows:
        print(f"{f(x)} {f(y)} {f(r)}")


if __name__ == "__main__":
    main()
