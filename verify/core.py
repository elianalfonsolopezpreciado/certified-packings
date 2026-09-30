"""Independent verifier for packing certificates.

This module deliberately imports NOTHING from the search code (only the standard library
and, for the high-precision backend, mpmath).

Certificate JSON formats (numbers may be JSON numbers or decimal strings; the *decimal text* is
what is certified, parsed exactly, never through binary floats):

  {"problem": "sum_radii",    "n": 26, "circles": [[x, y, r], ...]}
  {"problem": "min_distance", "n": 10, "points":  [[x, y], ...]}

Policy (sum_radii): the certificate is checked after shrinking every radius by EPS
(default 1e-12), r'_i = max(r_i - EPS, 0).  The shrunken configuration must satisfy, in EXACT
arithmetic (Fractions) or in 60-digit mpmath arithmetic, all constraints with violation <= 0:

    x_i - r'_i >= 0,  1 - x_i - r'_i >= 0,  (same for y)          (containment)
    (x_i-x_j)^2 + (y_i-y_j)^2 - (r'_i + r'_j)^2 >= 0               (non-overlap)

The certified score is sum_i r'_i (a genuinely feasible configuration), so a certified score
never over-states what the file supports.  The raw (unshrunken) violation is reported too.

Policy (min_distance): points must lie in [0,1]^2 exactly; the value is sqrt(min_{i<j} d_ij^2)
computed from the exact rational minimum squared distance.
"""
from __future__ import annotations

import json
import os

# Force mpmath's pure-Python integer backend: deterministic, and avoids probing gmpy2 (whose DLL is
# broken in some conda installs, printing a spurious fatal-exception trace).
os.environ.setdefault("MPMATH_NOGMPY", "1")

from dataclasses import dataclass, field  # noqa: E402
from fractions import Fraction  # noqa: E402
from typing import Any, Sequence  # noqa: E402

DEFAULT_EPS = Fraction(1, 10**12)
_MP_DPS = 60


class CertificateError(ValueError):
    """The certificate file is malformed (not merely infeasible)."""


# ----------------------------------------------------------------------------- parsing
def _to_fraction(v: Any) -> Fraction:
    """Exact conversion of a JSON number/decimal string to a Fraction."""
    if isinstance(v, bool):
        raise CertificateError(f"boolean is not a number: {v!r}")
    if isinstance(v, (int, str)):
        text = str(v).strip()
    else:
        raise CertificateError(f"unsupported number type: {type(v).__name__}")
    low = text.lower()
    if low in ("nan", "inf", "+inf", "-inf", "infinity", "-infinity") or "nan" in low or "inf" in low:
        raise CertificateError(f"non-finite number: {v!r}")
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise CertificateError(f"cannot parse number {v!r}") from exc


def load_certificate(path: str) -> dict[str, Any]:
    """Load JSON keeping the original decimal text of every float (parse_float=str)."""
    if str(path).endswith(".gz"):
        import gzip

        opener = lambda: gzip.open(path, "rt", encoding="utf-8")  # noqa: E731
    else:
        opener = lambda: open(path, "r", encoding="utf-8")  # noqa: E731
    with opener() as fh:
        data = json.load(fh, parse_float=str, parse_constant=_reject_constant)
    if not isinstance(data, dict):
        raise CertificateError("top-level JSON must be an object")
    return data


def _reject_constant(name: str) -> Any:
    raise CertificateError(f"non-finite JSON constant: {name}")


@dataclass
class Parsed:
    problem: str
    n: int
    rows: list[tuple[Fraction, ...]]


def parse(data: dict[str, Any]) -> Parsed:
    problem = data.get("problem")
    if problem == "sum_radii":
        key, width = "circles", 3
    elif problem == "min_distance":
        key, width = "points", 2
    elif problem == "autocorr1":
        key, width = "heights", 1
    else:
        raise CertificateError(f"unknown problem: {problem!r}")
    raw = data.get(key)
    if not isinstance(raw, list):
        raise CertificateError(f"missing list '{key}'")
    n_decl = data.get("n")
    if n_decl is not None and int(n_decl) != len(raw):
        raise CertificateError(f"declared n={n_decl} but {len(raw)} entries present")
    rows: list[tuple[Fraction, ...]] = []
    for i, item in enumerate(raw):
        if width == 1 and not isinstance(item, (list, dict)):
            item = [item]
        if isinstance(item, dict):
            names = ("x", "y", "r")[:width]
            if any(k not in item for k in names):
                raise CertificateError(f"entry {i} missing one of {names}")
            item = [item[k] for k in names]
        if not isinstance(item, list) or len(item) != width:
            raise CertificateError(f"entry {i} must have {width} numbers")
        rows.append(tuple(_to_fraction(v) for v in item))
    if not rows:
        raise CertificateError("empty certificate")
    return Parsed(problem, len(rows), rows)


# ----------------------------------------------------------------------------- results
@dataclass
class Report:
    problem: str
    n: int
    backend: str
    valid: bool
    certified_score: str  # decimal string, >= 20 significant digits
    raw_score: str
    max_violation: str  # largest constraint violation of the SHRUNKEN config (<=0 means feasible)
    raw_max_violation: str  # same for the unshrunken config
    eps: str
    n_violations: int = 0
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _fmt(x: Any, digits: int = 25) -> str:
    """Format a Fraction / mpf to a decimal string with `digits` significant digits."""
    import mpmath

    with mpmath.workdps(digits + 10):
        v = mpmath.mpf(x.numerator) / mpmath.mpf(x.denominator) if isinstance(x, Fraction) else mpmath.mpf(x)
        return mpmath.nstr(v, digits, strip_zeros=False)


# ----------------------------------------------------------------------------- Fraction backend
def _sum_radii_violation_fraction(rows: Sequence[tuple[Fraction, ...]], eps: Fraction):
    """Return (score, max_violation, n_violating_constraints, messages) in exact arithmetic.

    violation of a constraint g >= 0 is max(0, -g)... we return max of (-g) so that <=0 is feasible.
    """
    n = len(rows)
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    rs = [max(r[2] - eps, Fraction(0)) for r in rows]
    raw_neg = [r[2] < 0 for r in rows]
    msgs: list[str] = []
    worst: Fraction | None = None
    bad = 0

    def note(g: Fraction, what: str) -> None:
        nonlocal worst, bad
        v = -g
        if worst is None or v > worst:
            worst = v
        if g < 0:
            bad += 1
            if len(msgs) < 10:
                msgs.append(f"{what}: slack {float(g):.3e}")

    for i in range(n):
        if raw_neg[i]:
            bad += 1
            msgs.append(f"circle {i}: negative radius")
        note(xs[i] - rs[i], f"circle {i} left wall")
        note(1 - xs[i] - rs[i], f"circle {i} right wall")
        note(ys[i] - rs[i], f"circle {i} bottom wall")
        note(1 - ys[i] - rs[i], f"circle {i} top wall")
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            s = rs[i] + rs[j]
            note(dx * dx + dy * dy - s * s, f"circles {i},{j} overlap (squared-distance slack)")
    assert worst is not None
    return sum(rs, Fraction(0)), worst, bad, msgs


# ----------------------------------------------------------------------------- autocorr1 (problem C)
SMALL_N = 200  # below this the plain O(N^2) loops are used (simplest, fully transparent)


def _autocorr1_exact(rows, method: str = "auto"):
    """Exact ratio R = 2N * max_m c_m / (sum a)^2 for the step function with N equal pieces on
    [-1/4, 1/4] and heights a_i >= 0, where c_m = sum_{i+j=m} a_i a_j.

    f*f is piecewise linear with vertices at the values h*c_m (h = 1/(2N)), so max(f*f) = h*max_m c_m and
    int f = h*sum(a): R = max(f*f)/(int f)^2 = 2N max c / (sum a)^2.  Done in exact integers; for large N the
    convolution uses ``verify.ac1`` (Kronecker big-int or int64 limbs), both exact.
    method: 'direct' | 'kronecker' | 'limbs' | 'auto' (direct if N <= SMALL_N else kronecker).
    Returns (R as Fraction, list of violations)."""
    from . import ac1

    a = [r[0] for r in rows]
    bad = [f"height {i} negative" for i, v in enumerate(a) if v < 0]
    A, _ = ac1.to_integers(a)
    s = sum(A)
    if s <= 0:
        return None, bad + ["sum of heights is not positive"]
    n = len(A)
    if method == "auto":
        method = "direct" if n <= SMALL_N else "kronecker"
    best = {"direct": lambda: max(ac1.conv_direct(A)), "kronecker": lambda: ac1.max_kronecker(A),
            "limbs": lambda: ac1.max_limbs(A)}[method]()
    return Fraction(2 * n * best, s * s), bad


def _verify_autocorr1(parsed: Parsed, backend: str) -> Report:
    import mpmath

    from . import ac1

    n = parsed.n
    if backend.startswith("mpmath"):
        # different code path from the fraction backend: mpf convolution for small N, the int64-limb exact
        # convolution (no shared code with the Kronecker path) for large N; final ratio in 60-digit mpmath.
        a = [r[0] for r in parsed.rows]
        bad = [f"height {i} negative" for i, v in enumerate(a) if v < 0]
        if n <= SMALL_N:
            with mpmath.workdps(_MP_DPS):
                am = [mpmath.mpf(v.numerator) / mpmath.mpf(v.denominator) for v in a]
                s = mpmath.fsum(am)
                if not s > 0:
                    return Report(parsed.problem, n, backend, False, "nan", "nan", "1", "1", "0", len(bad) + 1,
                                  bad + ["sum of heights is not positive"])
                best = mpmath.mpf(0)
                for m in range(2 * n - 1):
                    lo, hi = max(0, m - n + 1), min(m, n - 1)
                    c = mpmath.fsum(am[i] * am[m - i] for i in range(lo, hi + 1))
                    if c > best:
                        best = c
                score = mpmath.nstr(2 * n * best / (s * s), 30, strip_zeros=False)
        else:
            A, _ = ac1.to_integers(a)
            s_int = sum(A)
            if s_int <= 0:
                return Report(parsed.problem, n, backend, False, "nan", "nan", "1", "1", "0", len(bad) + 1,
                              bad + ["sum of heights is not positive"])
            cmax = ac1.max_limbs(A)
            with mpmath.workdps(_MP_DPS):
                score = mpmath.nstr(mpmath.mpf(2 * n * cmax) / mpmath.mpf(s_int) ** 2, 30, strip_zeros=False)
        ok = not bad
        return Report(parsed.problem, n, backend, ok, score, score, "0" if ok else "1", "0" if ok else "1", "0",
                      len(bad), bad)
    R, bad = _autocorr1_exact(parsed.rows)
    if R is None:
        return Report(parsed.problem, n, backend, False, "nan", "nan", "1", "1", "0", len(bad), bad)
    score = _fmt(R, 30)
    return Report(parsed.problem, n, backend, len(bad) == 0, score, score, "0" if not bad else "1",
                  "0" if not bad else "1", "0", len(bad), bad)


def verify_fraction(parsed: Parsed, eps: Fraction = DEFAULT_EPS) -> Report:
    if parsed.problem == "autocorr1":
        return _verify_autocorr1(parsed, "fraction-exact-int")
    if parsed.problem == "sum_radii":
        score, worst, bad, msgs = _sum_radii_violation_fraction(parsed.rows, eps)
        raw_score, raw_worst, _, _ = _sum_radii_violation_fraction(parsed.rows, Fraction(0))
        return Report(parsed.problem, parsed.n, "fraction-exact", bad == 0, _fmt(score), _fmt(raw_score),
                      _fmt(worst), _fmt(raw_worst), _fmt(eps, 6), bad, msgs)
    # min_distance
    import mpmath

    pts = parsed.rows
    msgs = []
    bad = 0
    for i, (x, y) in enumerate(pts):
        if not (0 <= x <= 1 and 0 <= y <= 1):
            bad += 1
            msgs.append(f"point {i} outside unit square")
    min_sq: Fraction | None = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d2 = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2
            if min_sq is None or d2 < min_sq:
                min_sq = d2
    if min_sq is None:  # n == 1
        min_sq = Fraction(0)
        msgs.append("n=1: min distance undefined, reported 0")
    with mpmath.workdps(_MP_DPS):
        d = mpmath.sqrt(mpmath.mpf(min_sq.numerator) / mpmath.mpf(min_sq.denominator))
        dstr = mpmath.nstr(d, 30, strip_zeros=False)
    worst = _fmt(Fraction(0) if bad == 0 else Fraction(1))
    return Report(parsed.problem, parsed.n, "fraction-exact", bad == 0, dstr, dstr, worst, worst, "0", bad, msgs)


# ----------------------------------------------------------------------------- mpmath backend
def verify_mpmath(parsed: Parsed, eps: Fraction = DEFAULT_EPS) -> Report:
    import mpmath

    if parsed.problem == "autocorr1":
        return _verify_autocorr1(parsed, f"mpmath-{_MP_DPS}dps")
    mp = mpmath.mp
    old = mp.dps
    mp.dps = _MP_DPS
    try:
        def M(fr: Fraction):
            return mpmath.mpf(fr.numerator) / mpmath.mpf(fr.denominator)

        if parsed.problem == "sum_radii":
            def run(e: Fraction):
                E = M(e)
                xs = [M(r[0]) for r in parsed.rows]
                ys = [M(r[1]) for r in parsed.rows]
                rs = [max(M(r[2]) - E, mpmath.mpf(0)) for r in parsed.rows]
                n = len(xs)
                worst = mpmath.mpf("-inf")
                bad = 0
                msgs: list[str] = []
                for i in range(n):
                    if parsed.rows[i][2] < 0:
                        bad += 1
                        msgs.append(f"circle {i}: negative radius")
                    for name, g in (("left", xs[i] - rs[i]), ("right", 1 - xs[i] - rs[i]),
                                    ("bottom", ys[i] - rs[i]), ("top", 1 - ys[i] - rs[i])):
                        worst = max(worst, -g)
                        if g < 0:
                            bad += 1
                            if len(msgs) < 10:
                                msgs.append(f"circle {i} {name} wall: slack {mpmath.nstr(g, 5)}")
                for i in range(n):
                    for j in range(i + 1, n):
                        g = (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 - (rs[i] + rs[j]) ** 2
                        worst = max(worst, -g)
                        if g < 0:
                            bad += 1
                            if len(msgs) < 10:
                                msgs.append(f"circles {i},{j} overlap: slack {mpmath.nstr(g, 5)}")
                return mpmath.fsum(rs), worst, bad, msgs

            score, worst, bad, msgs = run(eps)
            raw_score, raw_worst, _, _ = run(Fraction(0))
            f = lambda v: mpmath.nstr(v, 25, strip_zeros=False)  # noqa: E731
            return Report(parsed.problem, parsed.n, f"mpmath-{_MP_DPS}dps", bad == 0, f(score), f(raw_score),
                          f(worst), f(raw_worst), _fmt(eps, 6), bad, msgs)

        pts = [(M(r[0]), M(r[1])) for r in parsed.rows]
        bad = 0
        msgs = []
        for i, (x, y) in enumerate(pts):
            if x < 0 or x > 1 or y < 0 or y > 1:
                bad += 1
                msgs.append(f"point {i} outside unit square")
        best = None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d2 = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2
                if best is None or d2 < best:
                    best = d2
        if best is None:
            best = mpmath.mpf(0)
        d = mpmath.sqrt(best)
        ds = mpmath.nstr(d, 30, strip_zeros=False)
        w = "0" if bad == 0 else "1"
        return Report(parsed.problem, parsed.n, f"mpmath-{_MP_DPS}dps", bad == 0, ds, ds, w, w, "0", bad, msgs)
    finally:
        mp.dps = old


BACKENDS = {"fraction": verify_fraction, "mpmath": verify_mpmath}


def verify_file(path: str, backend: str = "fraction", eps: Fraction = DEFAULT_EPS) -> Report:
    return BACKENDS[backend](parse(load_certificate(path)), eps)
