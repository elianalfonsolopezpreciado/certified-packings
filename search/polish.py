"""High-precision active-set Newton polish.

Given a float local optimum, identify the active constraints, then solve   g_A(z) = 0   in
mpmath (default 60 digits) by minimum-norm Newton / Gauss-Newton steps

        dz = - J^T (J J^T)^{-1} g            (J restricted to an independent row subset)

Circles / points that take part in no active constraint (rattlers) are frozen. The result
satisfies every active constraint to ~1e-50; all other constraints are re-checked in high precision
and, if needed, radii are shrunk by the largest violation. The KKT multipliers are estimated in
floating point and reported (`min_lambda >= 0`, small `kkt_residual` indicate a genuine local optimum).

Nothing here certifies anything: the output JSON is handed to the independent `verify` package.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPMATH_NOGMPY", "1")

import mpmath as mpm  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import qr as sqr  # noqa: E402

Row = tuple[str, tuple[int, ...]]  # (kind, indices)


def _float_rows_sr(z: np.ndarray, n: int, tol: float) -> list[Row]:
    x, y, r = z[:n], z[n : 2 * n], z[2 * n :]
    rows: list[Row] = []
    for i in range(n):
        if x[i] - r[i] < tol:
            rows.append(("wl", (i,)))
        if 1 - x[i] - r[i] < tol:
            rows.append(("wr", (i,)))
        if y[i] - r[i] < tol:
            rows.append(("wb", (i,)))
        if 1 - y[i] - r[i] < tol:
            rows.append(("wt", (i,)))
    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(x[i] - x[j], y[i] - y[j]) - (r[i] + r[j]) < tol:
                rows.append(("pair", (i, j)))
    return rows


def _eval_sr(kind, idx, Z, n):
    """Return (value, {var_index: derivative}) with Z an mp list (x..., y..., r...)."""
    if kind == "pair":
        i, j = idx
        dx, dy, s = Z[i] - Z[j], Z[n + i] - Z[n + j], Z[2 * n + i] + Z[2 * n + j]
        return dx * dx + dy * dy - s * s, {i: 2 * dx, j: -2 * dx, n + i: 2 * dy, n + j: -2 * dy,
                                            2 * n + i: -2 * s, 2 * n + j: -2 * s}
    (i,) = idx
    one = mpm.mpf(1)
    if kind == "wl":
        return Z[i] - Z[2 * n + i], {i: one, 2 * n + i: -one}
    if kind == "wr":
        return 1 - Z[i] - Z[2 * n + i], {i: -one, 2 * n + i: -one}
    if kind == "wb":
        return Z[n + i] - Z[2 * n + i], {n + i: one, 2 * n + i: -one}
    return 1 - Z[n + i] - Z[2 * n + i], {n + i: -one, 2 * n + i: -one}  # "wt"


def _float_rows_md(z: np.ndarray, n: int, tol: float):
    x, y = z[:n], z[n : 2 * n]
    ii, jj = np.triu_indices(n, 1)
    d = np.hypot(x[ii] - x[jj], y[ii] - y[jj])
    dmin = d.min()
    rows = [("pair", (int(i), int(j))) for i, j, dd in zip(ii, jj, d) if dd - dmin < tol]
    return rows, dmin


def _eval_md(kind, idx, Z, n):
    i, j = idx
    dx, dy = Z[i] - Z[j], Z[n + i] - Z[n + j]
    return dx * dx + dy * dy - Z[2 * n], {i: 2 * dx, j: -2 * dx, n + i: 2 * dy, n + j: -2 * dy,
                                          2 * n: -mpm.mpf(1)}


def _independent_rows(Jf: np.ndarray) -> list[int]:
    if Jf.shape[0] == 0:
        return []
    _, R, piv = sqr(Jf.T, pivoting=True, mode="economic")
    d = np.abs(np.diag(R))
    rank = int(np.sum(d > 1e-9 * max(d[0], 1e-300)))
    return sorted(int(p) for p in piv[:rank])


def polish(kind: str, n: int, z: np.ndarray, dps: int = 60, tol: float = 1e-8, iters: int = 30) -> dict:
    """Return {'certificate': dict(strings), 'info': {...}}."""
    z = np.array(z, dtype=float)
    mpm.mp.dps = dps
    if kind == "sum_radii":
        rows = _float_rows_sr(z, n, tol)
        evalf = _eval_sr
        Z = [mpm.mpf(float(v)) for v in z]
        nv = 3 * n
        frozen: set[int] = set()
    else:
        z = z.copy()
        # snap coordinates on the boundary exactly onto it (they become frozen constants)
        frozen = set()
        for k in range(2 * n):
            if z[k] < 1e-9:
                z[k] = 0.0
                frozen.add(k)
            elif z[k] > 1 - 1e-9:
                z[k] = 1.0
                frozen.add(k)
        rows, dmin = _float_rows_md(z, n, tol)
        z[2 * n] = dmin**2
        evalf = _eval_md
        Z = [mpm.mpf(float(v)) for v in z]
        nv = 2 * n + 1
    involved = sorted({v for kd, idx in rows for v in _eval_vars(kind, kd, idx, n)} - frozen)
    col = {v: c for c, v in enumerate(involved)}

    def build(Zc):
        g, Jrows = [], []
        for kd, idx in rows:
            val, der = evalf(kd, idx, Zc, n)
            g.append(val)
            Jrows.append({v: d for v, d in der.items() if v in col})
        return g, Jrows

    info = {"n_active": len(rows), "n_unknowns": len(involved)}
    g, Jr = build(Z)
    Jf = np.zeros((len(rows), len(involved)))
    for a, r in enumerate(Jr):
        for v, d in r.items():
            Jf[a, col[v]] = float(d)
    keep = _independent_rows(Jf)
    info["rank"] = len(keep)
    hist = []
    for it in range(iters):
        g, Jr = build(Z)
        res = max((abs(g[a]) for a in range(len(rows))), default=mpm.mpf(0))
        hist.append(float(res))
        if res < mpm.mpf(10) ** (-(dps - 12)):
            break
        m = len(keep)
        A = mpm.matrix(m, m)
        for p, a in enumerate(keep):
            for q, b in enumerate(keep):
                if q < p:
                    A[p, q] = A[q, p]
                    continue
                A[p, q] = mpm.fsum(d * Jr[b][v] for v, d in Jr[a].items() if v in Jr[b])
        rhs = mpm.matrix([g[a] for a in keep])
        y = mpm.lu_solve(A, rhs)
        for p, a in enumerate(keep):
            for v, d in Jr[a].items():
                Z[v] = Z[v] - d * y[p]
    info["newton_residuals"] = hist
    info["max_active_residual"] = hist[-1] if hist else 0.0

    # ---- final high-precision full check; shrink radii by the largest violation if any ----
    delta = mpm.mpf(0)
    if kind == "sum_radii":
        worst = mpm.mpf(0)
        for i in range(n):
            for gval in (Z[i] - Z[2 * n + i], 1 - Z[i] - Z[2 * n + i], Z[n + i] - Z[2 * n + i], 1 - Z[n + i] - Z[2 * n + i]):
                worst = max(worst, -gval)
        for i in range(n):
            for j in range(i + 1, n):
                dist = mpm.sqrt((Z[i] - Z[j]) ** 2 + (Z[n + i] - Z[n + j]) ** 2)
                worst = max(worst, (Z[2 * n + i] + Z[2 * n + j]) - dist)
        delta = worst if worst > 0 else mpm.mpf(0)
        if delta > 0:
            for i in range(n):
                Z[2 * n + i] -= delta
        info["shrink_applied"] = float(delta)
        cert = {"problem": "sum_radii", "n": n,
                "circles": [[mpm.nstr(Z[i], 40, strip_zeros=False), mpm.nstr(Z[n + i], 40, strip_zeros=False),
                             mpm.nstr(Z[2 * n + i], 40, strip_zeros=False)] for i in range(n)]}
        info["score_mp"] = mpm.nstr(mpm.fsum(Z[2 * n : 3 * n]), 30)
    else:
        cert = {"problem": "min_distance", "n": n,
                "points": [[mpm.nstr(Z[i], 40, strip_zeros=False), mpm.nstr(Z[n + i], 40, strip_zeros=False)]
                           for i in range(n)]}
        info["score_mp"] = mpm.nstr(mpm.sqrt(Z[2 * n]), 30)
    info["score_float_in"] = float(np.sum(z[2 * n :])) if kind == "sum_radii" else float(np.sqrt(z[2 * n]))

    # ---- KKT multipliers (float): c + J^T lambda = 0, lambda >= 0 for a maximiser of -f ----
    try:
        gvec = np.zeros(len(involved))
        if kind == "sum_radii":
            for v in involved:
                if v >= 2 * n:
                    gvec[col[v]] = 1.0  # gradient of sum r  (we maximise)
        else:
            gvec[col[2 * n]] = 1.0  # maximise t
        # at a maximiser of f subject to g >= 0:  grad f + sum lambda_k grad g_k = 0 with lambda >= 0
        lam, *_ = np.linalg.lstsq(Jf.T, -gvec, rcond=None)
        info["kkt_residual"] = float(np.max(np.abs(Jf.T @ lam + gvec))) if len(lam) else float(np.max(np.abs(gvec)))
        info["min_lambda"] = float(lam.min()) if len(lam) else 0.0
    except Exception as exc:  # pragma: no cover
        info["kkt_error"] = str(exc)
    return {"certificate": cert, "info": info}


def _eval_vars(kind: str, kd: str, idx: tuple[int, ...], n: int) -> list[int]:
    if kind == "sum_radii":
        if kd == "pair":
            i, j = idx
            return [i, j, n + i, n + j, 2 * n + i, 2 * n + j]
        (i,) = idx
        return [i if kd in ("wl", "wr") else n + i, 2 * n + i]
    i, j = idx
    return [i, j, n + i, n + j, 2 * n]


# =============================================================================================
# Mixed-precision variant for large n (sum_radii only)
# =============================================================================================
def polish_mixed(n: int, z: np.ndarray, dps: int = 60, tol: float = 1e-8, iters: int = 30) -> dict:
    """Same result format as `polish`, but each Newton step solves the linear system in float64 while the
    residual g(z) is evaluated in `dps`-digit mpmath (iterative refinement: every step gains ~8-13 digits).
    Cost is O(m) mp operations per step instead of the O(m^3) mp LU of `polish`, so n ~ 300 takes seconds."""
    from .large import neighbor_pairs

    z = np.array(z, dtype=float)
    mpm.mp.dps = dps
    x, y, r = z[:n], z[n : 2 * n], z[2 * n :]
    rows: list[Row] = []
    for i in range(n):
        if x[i] - r[i] < tol:
            rows.append(("wl", (i,)))
        if 1 - x[i] - r[i] < tol:
            rows.append(("wr", (i,)))
        if y[i] - r[i] < tol:
            rows.append(("wb", (i,)))
        if 1 - y[i] - r[i] < tol:
            rows.append(("wt", (i,)))
    I, J = neighbor_pairs(x, y, r, tol)
    for i, j in zip(I.tolist(), J.tolist()):
        rows.append(("pair", (i, j)))
    Z = [mpm.mpf(float(v)) for v in z]
    involved = sorted({v for kd, idx in rows for v in _eval_vars("sum_radii", kd, idx, n)})
    col = {v: c for c, v in enumerate(involved)}
    info: dict = {"n_active": len(rows), "n_unknowns": len(involved), "mode": "mixed-precision"}

    def resid_and_jac(Zc, want_j: bool):
        g = []
        Jf = np.zeros((len(rows), len(involved))) if want_j else None
        for a, (kd, idx) in enumerate(rows):
            val, der = _eval_sr(kd, idx, Zc, n)
            g.append(val)
            if want_j:
                for v, d in der.items():
                    Jf[a, col[v]] = float(d)
        return g, Jf

    g, Jf = resid_and_jac(Z, True)
    keep = _independent_rows(Jf)
    info["rank"] = len(keep)
    hist = []
    for it in range(iters):
        if it:
            g, Jf = resid_and_jac(Z, True)
        res = max((abs(v) for v in g), default=mpm.mpf(0))
        hist.append(float(res))
        if res < mpm.mpf(10) ** (-(dps - 12)):
            break
        gk = [g[a] for a in keep]
        s = max(abs(v) for v in gk)
        gf = np.array([float(v / s) for v in gk])
        Jk = Jf[keep]
        y_ = np.linalg.solve(Jk @ Jk.T + 1e-300 * np.eye(len(keep)), gf)
        dz = Jk.T @ y_  # float64; scaled by s below in mp
        sm = mpm.mpf(s)
        for v, c in col.items():
            Z[v] = Z[v] - sm * mpm.mpf(float(dz[c]))
    info["newton_residuals"] = hist
    info["max_active_residual"] = hist[-1] if hist else 0.0

    # final exact-ish full check in mp (all pairs) + shrink by the largest violation
    Xm, Ym, Rm = Z[:n], Z[n : 2 * n], Z[2 * n :]
    worst = mpm.mpf(0)
    for i in range(n):
        for gv in (Xm[i] - Rm[i], 1 - Xm[i] - Rm[i], Ym[i] - Rm[i], 1 - Ym[i] - Rm[i]):
            worst = max(worst, -gv)
    xs, ys, rs = np.array([float(v) for v in Xm]), np.array([float(v) for v in Ym]), np.array([float(v) for v in Rm])
    I2, J2 = neighbor_pairs(xs, ys, rs, 1e-3)  # any pair further apart than 1e-3 in gap cannot be violated
    for i, j in zip(I2.tolist(), J2.tolist()):
        dist = mpm.sqrt((Xm[i] - Xm[j]) ** 2 + (Ym[i] - Ym[j]) ** 2)
        worst = max(worst, (Rm[i] + Rm[j]) - dist)
    info["full_check_pairs"] = int(len(I2))
    delta = worst if worst > 0 else mpm.mpf(0)
    if delta > 0:
        for i in range(n):
            Z[2 * n + i] -= delta
    info["shrink_applied"] = float(delta)
    cert = {"problem": "sum_radii", "n": n,
            "circles": [[mpm.nstr(Z[i], 40, strip_zeros=False), mpm.nstr(Z[n + i], 40, strip_zeros=False),
                         mpm.nstr(Z[2 * n + i], 40, strip_zeros=False)] for i in range(n)]}
    info["score_mp"] = mpm.nstr(mpm.fsum(Z[2 * n : 3 * n]), 30)
    info["score_float_in"] = float(np.sum(z[2 * n :]))
    try:
        gvec = np.zeros(len(involved))
        for v in involved:
            if v >= 2 * n:
                gvec[col[v]] = 1.0
        lam, *_ = np.linalg.lstsq(Jf.T, -gvec, rcond=None)
        info["kkt_residual"] = float(np.max(np.abs(Jf.T @ lam + gvec)))
        info["min_lambda"] = float(lam.min())
    except Exception as exc:  # pragma: no cover
        info["kkt_error"] = str(exc)
    return {"certificate": cert, "info": info}
