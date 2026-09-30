"""Diagonal-mirror (D1) symmetric search for Problem A.

A symmetric packing under the reflection (x, y) -> (y, x) consists of p free circles (each with its mirror image) and
q circles centred on the diagonal (x = y), n = 2p + q.  Reduced variables:

    zr = [x_f(p), y_f(p), r_f(p), t_d(q), r_d(q)]

Full configuration:  x = [x_f, y_f, t_d],  y = [y_f, x_f, t_d],  r = [r_f, r_f, r_d].
The map zr -> z is LINEAR, z = M zr (constant sparse matrix M for x, y, r blocks), so the sparse SLP of search/large.py can be
run in reduced variables (A_red = A M, c_red = M^T c) with the same exact neighbour-list argument.  Symmetric optima are
also local optima candidates of the unconstrained problem; the caller relaxes them with the full SLP afterwards.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
from scipy.optimize import linprog  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402

from .large import TR_FACTOR, _feasible_radii, neighbor_pairs  # noqa: E402


def nred(p: int, q: int) -> int:
    return 3 * p + 2 * q


def expand(zr: np.ndarray, p: int, q: int) -> np.ndarray:
    xf, yf, rf = zr[:p], zr[p : 2 * p], zr[2 * p : 3 * p]
    td, rd = zr[3 * p : 3 * p + q], zr[3 * p + q :]
    x = np.concatenate([xf, yf, td])
    y = np.concatenate([yf, xf, td])
    r = np.concatenate([rf, rf, rd])
    return np.concatenate([x, y, r])


def reduce_z(z: np.ndarray, p: int, q: int) -> np.ndarray:
    """Inverse of expand for an (approximately) symmetric z laid out as in expand."""
    n = 2 * p + q
    x, y, r = z[:n], z[n : 2 * n], z[2 * n :]
    return np.concatenate([x[:p], y[:p], r[:p], 0.5 * (x[2 * p :] + y[2 * p :]), r[2 * p :]])


def sub_matrix(p: int, q: int):
    """Sparse M (3n x nred) with z = M zr."""
    n = 2 * p + q
    rows, cols = [], []
    k = np.arange(p)
    j = np.arange(q)
    # x block rows 0..n-1
    rows += [k, p + k, 2 * p + j]
    cols += [k, p + k, 3 * p + j]
    # y block rows n..2n-1
    rows += [n + k, n + p + k, n + 2 * p + j]
    cols += [p + k, k, 3 * p + j]
    # r block rows 2n..3n-1
    rows += [2 * n + k, 2 * n + p + k, 2 * n + 2 * p + j]
    cols += [2 * p + k, 2 * p + k, 3 * p + q + j]
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(3 * n, nred(p, q))).tocsr()


def slp_step_sym(x, y, r, delta: float, M, p: int, q: int):
    """One symmetric LP step in reduced variables. Returns delta_full (3n) or None."""
    n = len(x)
    margin = TR_FACTOR * delta
    I, J = neighbor_pairs(x, y, r, margin)
    m = len(I)
    rows, cols, vals, rhs = [], [], [], []
    idx = np.arange(n)
    for k, (cx, sgn, base) in enumerate(((0, -1.0, x - r), (0, 1.0, 1 - x - r), (1, -1.0, y - r), (1, 1.0, 1 - y - r))):
        row = k * n + idx
        rows += [row, row]
        cols += [cx * n + idx, 2 * n + idx]
        vals += [np.full(n, sgn), np.ones(n)]
        rhs.append(base)
    if m:
        dxp, dyp = x[I] - x[J], y[I] - y[J]
        d = np.hypot(dxp, dyp)
        ux, uy = dxp / np.maximum(d, 1e-300), dyp / np.maximum(d, 1e-300)
        row = 4 * n + np.arange(m)
        rows += [row] * 6
        cols += [I, J, n + I, n + J, 2 * n + I, 2 * n + J]
        vals += [-ux, ux, -uy, uy, np.ones(m), np.ones(m)]
        rhs.append(d - (r[I] + r[J]))
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(4 * n + m, 3 * n)).tocsr()
    Ar = (A @ M).tocsr()
    b = np.concatenate(rhs)
    cfull = np.zeros(3 * n)
    cfull[2 * n :] = -1.0
    cr = np.asarray(M.T @ cfull).ravel()
    nr = M.shape[1]
    # bounds on reduced variables: coordinates +-delta; radii in [max(-r, -delta), delta]
    zr_r = np.concatenate([r[:p], r[2 * p :]])
    lb = np.full(nr, -delta)
    ub = np.full(nr, delta)
    lb[2 * p : 3 * p] = np.maximum(-r[:p], -delta)
    lb[3 * p + q :] = np.maximum(-r[2 * p :], -delta)
    res = linprog(cr, A_ub=Ar, b_ub=b, bounds=np.stack([lb, ub], 1), method="highs",
                  options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
    if res.status != 0:
        return None
    return M @ res.x


def slp_local_sym(zr: np.ndarray, p: int, q: int, max_iter: int = 200, delta0: float = 0.03, tol: float = 1e-12,
                  patience: int = 3):
    """Symmetric sparse SLP. Returns (zr, iterations); every iterate is a symmetric feasible packing."""
    n = 2 * p + q
    M = sub_matrix(p, q)
    z = expand(zr, p, q)
    x, y, r = z[:n].copy(), z[n : 2 * n].copy(), z[2 * n :].copy()
    r = _feasible_radii(x, y, r)
    zr = reduce_z(np.concatenate([x, y, r]), p, q)
    delta, stall, total, it = delta0, 0, float(r.sum()), 0
    for it in range(1, max_iter + 1):
        z = expand(zr, p, q)
        x, y, r = z[:n], z[n : 2 * n], z[2 * n :]
        st = slp_step_sym(x, y, r, delta, M, p, q)
        if st is None:
            delta *= 0.5
            if delta < 1e-12:
                break
            continue
        znew = z + st
        new = float(znew[2 * n :].sum())
        gain = new - total
        if gain < -1e-9:
            delta *= 0.5
            continue
        zr = reduce_z(znew, p, q)
        total = new
        step = float(np.abs(st).max())
        delta = float(np.clip(3.0 * step, 1e-9, 0.05))
        if gain < tol:
            stall += 1
            if stall >= patience:
                break
        else:
            stall = 0
    return zr, it
