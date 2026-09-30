"""Large-n machinery for Problem A (sum of radii): sparse SLP with neighbour lists.

Variables z = [x(n), y(n), r(n)].  One SLP step solves the LP (HiGHS, sparse)

    max  sum(dr)   s.t.  |dx|,|dy|,|dr| <= Delta,  r + dr >= 0,
         walls   (linear, exact):        x+dx-(r+dr) >= 0,  1-(x+dx)-(r+dr) >= 0,   same for y
         pairs   (i,j) in neighbour list: d_ij + u_ij.(d_i - d_j) - (r_i+dr_i) - (r_j+dr_j) >= 0

with u_ij the unit vector from j to i.  Because the Euclidean distance is CONVEX in the centres,
d(z+delta) >= d + u.delta, hence any LP-feasible step is truly feasible: iterates stay feasible (up to the
LP tolerance) and sum(r) increases monotonically.  A pair whose gap d - (r_i + r_j) exceeds
2*sqrt(2)*Delta + 2*Delta cannot become violated inside the trust region, so restricting the LP to the neighbour
list of pairs with a smaller gap is EXACT for that step (not a heuristic).
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
from scipy.optimize import linprog  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

TR_FACTOR = 2.0 * np.sqrt(2.0) + 2.0  # gap beyond which a pair cannot become active within a step of size Delta


def neighbor_pairs(x: np.ndarray, y: np.ndarray, r: np.ndarray, margin: float):
    """All pairs (i<j) with d_ij - (r_i + r_j) < margin (cKDTree query, then exact filter)."""
    pts = np.stack([x, y], 1)
    rmax = float(r.max()) if len(r) else 0.0
    pairs = cKDTree(pts).query_pairs(2.0 * rmax + margin + 1e-12, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.hypot(x[i] - x[j], y[i] - y[j])
    keep = d - (r[i] + r[j]) < margin
    return i[keep], j[keep]


def all_pairs_max_violation(x, y, r) -> float:
    """Brute-force (all pairs, chunked) max of (r_i + r_j - d_ij) and wall violations; <= 0 means feasible."""
    n = len(x)
    worst = float(max(np.max(r - x), np.max(x + r - 1), np.max(r - y), np.max(y + r - 1), np.max(-r)))
    step = max(1, int(4e6 // max(n, 1)))
    for a in range(0, n, step):
        b = min(n, a + step)
        d = np.hypot(x[a:b, None] - x[None, :], y[a:b, None] - y[None, :])
        s = r[a:b, None] + r[None, :]
        v = s - d
        idx = np.arange(a, b)
        v[np.arange(b - a), idx] = -1.0  # ignore i == j
        worst = max(worst, float(v.max()))
    return worst


def slp_step(x, y, r, delta: float):
    """One LP step. Returns (dx, dy, dr) or None if the LP failed."""
    n = len(x)
    margin = TR_FACTOR * delta
    I, J = neighbor_pairs(x, y, r, margin)
    m = len(I)
    nv = 3 * n
    rows, cols, vals, rhs = [], [], [], []
    idx = np.arange(n)
    # walls: -dx + dr <= x - r ; dx + dr <= 1 - x - r ; same for y
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
        # -u.(d_i - d_j) + dr_i + dr_j <= d - (r_i + r_j)
        rows += [row] * 6
        cols += [I, J, n + I, n + J, 2 * n + I, 2 * n + J]
        vals += [-ux, ux, -uy, uy, np.ones(m), np.ones(m)]
        rhs.append(d - (r[I] + r[J]))
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(4 * n + m, nv)).tocsr()
    b = np.concatenate(rhs)
    c = np.zeros(nv)
    c[2 * n :] = -1.0
    lb = np.concatenate([np.full(2 * n, -delta), np.maximum(-r, -delta)])
    ub = np.full(nv, delta)
    res = linprog(c, A_ub=A, b_ub=b, bounds=np.stack([lb, ub], 1), method="highs",
                  options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
    if res.status != 0:
        return None
    s = res.x
    return s[:n], s[n : 2 * n], s[2 * n :]


def slp_local(z: np.ndarray, n: int, max_iter: int = 200, delta0: float = 0.03, tol: float = 1e-12, patience: int = 3):
    """Sparse SLP from z (feasible or nearly so). Returns (z, iterations)."""
    x, y, r = (np.array(z[:n], float), np.array(z[n : 2 * n], float), np.array(z[2 * n :], float))
    # make the start feasible by shrinking radii (exact LP on the centres would also do; this is cheap and safe)
    r = _feasible_radii(x, y, r)
    delta = delta0
    stall = 0
    total = float(r.sum())
    it = 0
    for it in range(1, max_iter + 1):
        st = slp_step(x, y, r, delta)
        if st is None:
            delta *= 0.5
            if delta < 1e-12:
                break
            continue
        dx, dy, dr = st
        x2, y2, r2 = x + dx, y + dy, r + dr
        new = float(r2.sum())
        gain = new - total
        if gain < -1e-9:  # LP tolerance issue: reject and shrink
            delta *= 0.5
            continue
        x, y, r, total = x2, y2, r2, new
        step = max(float(np.abs(dx).max()), float(np.abs(dy).max()), float(np.abs(dr).max()))
        delta = float(np.clip(3.0 * step, 1e-9, 0.05))
        if gain < tol:
            stall += 1
            if stall >= patience:
                break
        else:
            stall = 0
    return np.concatenate([x, y, r]), it


def _feasible_radii(x, y, r):
    """Shrink radii (largest violations first, simple iterative scaling) so every constraint holds to ~1e-15."""
    r = np.minimum(r, np.minimum.reduce([x, 1 - x, y, 1 - y]))
    r = np.maximum(r, 0.0)
    n = len(x)
    for _ in range(50):
        I, J = neighbor_pairs(x, y, r, 1e-6)
        if len(I) == 0:
            break
        d = np.hypot(x[I] - x[J], y[I] - y[J])
        v = r[I] + r[J] - d
        bad = v > 0
        if not bad.any():
            break
        Ib, Jb, vb = I[bad], J[bad], v[bad]
        # shrink both circles of every violating pair by half the violation (+ tiny margin)
        cut = np.zeros(n)
        np.maximum.at(cut, Ib, vb / 2 + 1e-15)
        np.maximum.at(cut, Jb, vb / 2 + 1e-15)
        r = np.maximum(r - cut, 0.0)
    return r
