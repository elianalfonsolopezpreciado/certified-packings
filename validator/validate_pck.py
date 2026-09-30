#!/usr/bin/env python3
"""Stand-alone validator for Packomania-style `.pck` files (variable-radius circles in a square, csqv).

Written from scratch as a third-party check: it uses ONLY the Python standard library (decimal, fractions, argparse, sys) and does not
import anything from this repository (neither `verify/` nor `search/`).

File layout (http://www.packomania.com/hints.html, read 2026-09-28 / 2026-09-29):
    line 1   radius of the largest circle (a plain number)
    line 2   author name(s)
    line 3.. `x y r` per circle, whitespace separated, sorted by INCREASING radius,
             container = square of side 1 centred at the origin, i.e. |x| + r <= 1/2 and |y| + r <= 1/2.

Checks, all in EXACT integer arithmetic on the decimal text (nothing is rounded, tolerance ZERO):
    1. every number parses as a finite decimal; r > 0
    2. every circle lies inside the square:   |x| + r <= 1/2,  |y| + r <= 1/2
    3. every pair is non-overlapping:         (dx)^2 + (dy)^2 >= (r_i + r_j)^2     (all n(n-1)/2 pairs)
    4. line 1 equals the largest radius and the rows are sorted by increasing radius (format checks)
    5. the sum of radii is recomputed exactly and printed with >= 15 significant digits
    6. informational: number of contacts within a tolerance (default 3e-12, Packomania's csqv rule asks for >= 3N contacts)

Exit status 0 iff checks 1-4 pass with zero tolerance (contact count is reported and, with --require-contacts, also enforced).

    python validator/validate_pck.py submission/csqv120.pck [--require-contacts] [--tol 3e-12]
"""
from __future__ import annotations

import argparse
import sys
from decimal import Decimal, getcontext
from fractions import Fraction

getcontext().prec = 60


def _parse_decimal(tok: str) -> Decimal:
    d = Decimal(tok)  # raises InvalidOperation on garbage
    if not d.is_finite():
        raise ValueError(f"non-finite number: {tok!r}")
    return d


def load(path: str):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) < 3:
        raise ValueError("file has fewer than 3 lines")
    largest = _parse_decimal(lines[0].strip())
    author = lines[1].strip()
    rows = []
    for i, ln in enumerate(lines[2:], start=3):
        tok = ln.split()
        if len(tok) != 3:
            raise ValueError(f"line {i}: expected 'x y r', got {len(tok)} columns")
        rows.append(tuple(_parse_decimal(t) for t in tok))
    return largest, author, rows


def to_ints(rows, largest):
    """Scale every number by 10^D (D = maximum number of decimals) so all arithmetic is on Python ints."""
    nums = [v for r in rows for v in r] + [largest]
    D = 0
    for v in nums:
        exp = v.as_tuple().exponent
        if isinstance(exp, int) and exp < 0:
            D = max(D, -exp)
    scale = Decimal(10) ** D
    conv = lambda v: int((v * scale).to_integral_exact())  # noqa: E731  (exact: v*10^D is an integer)
    for v in nums:
        assert v * scale == (v * scale).to_integral_exact()
    return [tuple(conv(v) for v in r) for r in rows], conv(largest), 10**D


def validate(path: str, tol: str = "3e-12", require_contacts: bool = False) -> dict:
    largest, author, rows = load(path)
    n = len(rows)
    ints, largest_i, S = to_ints(rows, largest)
    problems: list[str] = []
    half = S // 2 if S % 2 == 0 else None
    if half is None:  # S = 10^D is even for D >= 1
        raise ValueError("need at least one decimal place")
    # 1-2: positivity and containment  |x| + r <= 1/2
    for i, (x, y, r) in enumerate(ints):
        if r <= 0:
            problems.append(f"circle {i + 1}: non-positive radius")
        if abs(x) + r > half:
            problems.append(f"circle {i + 1}: leaves the square in x by {(abs(x) + r - half) / S:.3e}")
        if abs(y) + r > half:
            problems.append(f"circle {i + 1}: leaves the square in y by {(abs(y) + r - half) / S:.3e}")
    # 3: exact zero-tolerance non-overlap over all pairs
    tol_i = int((Fraction(tol) * S).__ceil__())
    contacts = 0
    worst = None  # smallest slack of squared distance (may be negative)
    for i in range(n):
        xi, yi, ri = ints[i]
        for j in range(i + 1, n):
            xj, yj, rj = ints[j]
            d2 = (xi - xj) ** 2 + (yi - yj) ** 2
            s2 = (ri + rj) ** 2
            if d2 < s2:
                problems.append(f"circles {i + 1},{j + 1} overlap (squared-distance deficit {(s2 - d2) / S**2:.3e})")
            if d2 <= (ri + rj + tol_i) ** 2:
                contacts += 1
            if worst is None or d2 - s2 < worst:
                worst = d2 - s2
    for x, y, r in ints:  # wall contacts within tolerance
        contacts += int(half - abs(x) - r <= tol_i) + int(half - abs(y) - r <= tol_i)
    # 4: format checks
    if largest_i != max(r for _, _, r in ints):
        problems.append("line 1 is not the largest radius")
    radii = [r for _, _, r in ints]
    if radii != sorted(radii):
        problems.append("rows are not sorted by increasing radius")
    total = sum(Decimal(r) for r in radii) / Decimal(S)  # exact: integer sum / 10^D
    if require_contacts and contacts < 3 * n:
        problems.append(f"only {contacts} contacts within {tol} (< 3N = {3 * n})")
    return {"path": path, "author": author, "n": n, "valid": not problems, "problems": problems, "sum_radii": total,
            "contacts_within_tol": contacts, "three_n": 3 * n, "tol": tol,
            "min_pair_slack": None if worst is None else Decimal(worst) / Decimal(S) ** 2}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pck", nargs="+")
    ap.add_argument("--tol", default="3e-12")
    ap.add_argument("--require-contacts", action="store_true")
    a = ap.parse_args(argv)
    ok = True
    for p in a.pck:
        try:
            rep = validate(p, a.tol, a.require_contacts)
        except Exception as exc:  # malformed file
            print(f"{p}: INVALID (cannot parse: {exc})")
            ok = False
            continue
        print(f"{p}: n={rep['n']} valid={rep['valid']} sum_of_radii={rep['sum_radii']:.20f} "
              f"contacts(<= {rep['tol']})={rep['contacts_within_tol']} (3N={rep['three_n']}) "
              f"min_pair_squared_slack={rep['min_pair_slack']:.3e}")
        for msg in rep["problems"][:10]:
            print("   !", msg)
        ok &= rep["valid"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
