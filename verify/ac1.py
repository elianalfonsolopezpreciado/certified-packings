"""Exact maximum of the autoconvolution of an integer sequence, for large N (problem C).

For heights a_i >= 0 (exact rationals) scale by the common denominator D so A_i = a_i * D are integers.  Then

    c_m = sum_{i+j=m} A_i A_j          (m = 0 .. 2N-2)   is an EXACT integer,
    R   = 2N * max_m c_m / (sum A)^2   is an EXACT rational upper bound on C_1.

Two independent exact algorithms (no floating-point FFT anywhere, so nothing to round):

* ``conv_max_kronecker`` - Kronecker substitution: pack A into one huge Python int with slot width B bits
  (B > bit_length(N * max(A)^2)), square it with Python's exact big-integer multiplication, unpack the
  slots.  Exact because no slot can carry into the next one.
* ``conv_max_limbs`` - split every A_i into 20-bit limbs, convolve limb sequences with numpy int64
  ``np.convolve`` (direct, exact: every partial sum < 2^63 for N <= 2^22), recombine limbs with Python ints.

They share no code; the certificate verifier uses one per backend and cross-checks them in tests.
Only the standard library and numpy are used (the search code is never imported).
"""
from __future__ import annotations

from fractions import Fraction
from math import lcm
from typing import Sequence

LIMB_BITS = 20


def to_integers(heights: Sequence[Fraction]) -> tuple[list[int], int]:
    """Return (A, D) with A_i = heights_i * D exact integers (D = lcm of denominators)."""
    D = 1
    for v in heights:
        D = lcm(D, v.denominator)
    return [int(v * D) for v in heights], D


def conv_direct(A: Sequence[int]) -> list[int]:
    """Reference O(N^2) exact autoconvolution (used for small N and in tests)."""
    n = len(A)
    out = []
    for m in range(2 * n - 1):
        lo, hi = max(0, m - n + 1), min(m, n - 1)
        c = 0
        for i in range(lo, hi + 1):
            c += A[i] * A[m - i]
        out.append(c)
    return out


def conv_kronecker(A: Sequence[int]) -> list[int]:
    n = len(A)
    if n == 0:
        return []
    bound = n * max(A) ** 2  # every c_m <= bound
    slot = (bound.bit_length() + 8) // 8 + 1  # bytes per slot, with slack
    buf = b"".join(int(a).to_bytes(slot, "little") for a in A)
    X = int.from_bytes(buf, "little")
    Y = X * X
    total = (2 * n - 1) * slot
    raw = Y.to_bytes(total + slot, "little")
    return [int.from_bytes(raw[m * slot : (m + 1) * slot], "little") for m in range(2 * n - 1)]


def conv_limbs(A: Sequence[int]) -> list[int]:
    import numpy as np

    n = len(A)
    if n == 0:
        return []
    if n > 2**22:
        raise ValueError("N too large for exact int64 limb sums")
    mask = (1 << LIMB_BITS) - 1
    nl = max((max(A).bit_length() + LIMB_BITS - 1) // LIMB_BITS, 1)
    limbs = [np.array([(a >> (LIMB_BITS * p)) & mask for a in A], dtype=np.int64) for p in range(nl)]
    # group by shift s = p + q: T_s[m] = sum_{p+q=s} (limb_p * limb_q)[m]
    T: dict[int, "np.ndarray"] = {}
    for p in range(nl):
        for q in range(p, nl):
            cv = np.convolve(limbs[p], limbs[q])
            if p != q:
                cv = cv * 2
            T[p + q] = T[p + q] + cv if (p + q) in T else cv
    # int64 safety: each term <= n * 2^(2*LIMB_BITS) * 2, at most nl terms per shift
    assert n * (1 << (2 * LIMB_BITS)) * 2 * nl < 2**63
    out = [0] * (2 * n - 1)
    for s, arr in T.items():
        vals = arr.tolist()
        sh = LIMB_BITS * s
        for m, v in enumerate(vals):
            out[m] += v << sh
    return out


def max_kronecker(A: Sequence[int]) -> int:
    return max(conv_kronecker(A))


def max_limbs(A: Sequence[int]) -> int:
    return max(conv_limbs(A))
