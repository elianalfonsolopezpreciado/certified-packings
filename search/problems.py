"""Problem definitions (variables, constraints, analytic Jacobians, local solvers).

Problem A  `SumRadii(n)`   : z = [x(n), y(n), r(n)],  maximise sum(r)
Problem B  `MinDistance(n)`: z = [x(n), y(n), t],      maximise t = (min pairwise distance)^2

Everything here is floating point and only used for *search*. Certification is done exclusively by
the independent `verify` package.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
from scipy.optimize import linprog, minimize  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402


@dataclass
class LocalResult:
    z: np.ndarray
    score: float  # feasible objective (radii repaired via exact LP / true min distance)
    raw_violation: float  # max constraint violation of the raw optimiser output
    nit: int


# =============================================================================================
# Problem A: sum of radii
# =============================================================================================
class SumRadii:
    kind = "sum_radii"

    def __init__(self, n: int):
        self.n = n
        self.dim = 3 * n
        self.iu = np.triu_indices(n, 1)
        self.bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
        n_ = n
        lin = np.zeros((4 * n_, 3 * n_))
        idx = np.arange(n_)
        lin[idx, idx] = 1.0
        lin[idx, 2 * n_ + idx] = -1.0  # x - r >= 0
        lin[n_ + idx, idx] = -1.0
        lin[n_ + idx, 2 * n_ + idx] = -1.0  # 1 - x - r >= 0
        lin[2 * n_ + idx, n_ + idx] = 1.0
        lin[2 * n_ + idx, 2 * n_ + idx] = -1.0  # y - r >= 0
        lin[3 * n_ + idx, n_ + idx] = -1.0
        lin[3 * n_ + idx, 2 * n_ + idx] = -1.0  # 1 - y - r >= 0
        self._lin = lin
        self._lin_c = np.concatenate([np.zeros(n_), np.ones(n_), np.zeros(n_), np.ones(n_)])
        self._grad = np.concatenate([np.zeros(2 * n_), -np.ones(n_)])

    # ---- helpers -------------------------------------------------------------------------
    def split(self, z):
        n = self.n
        return z[:n], z[n : 2 * n], z[2 * n :]

    def score_raw(self, z) -> float:
        return float(np.sum(z[2 * self.n :]))

    def max_violation(self, z) -> float:
        x, y, r = self.split(z)
        v = max(0.0, float(np.max(-(x - r))), float(np.max(-(1 - x - r))), float(np.max(-(y - r))),
                float(np.max(-(1 - y - r))), float(np.max(-r)))
        i, j = self.iu
        g = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 - (r[i] + r[j]) ** 2
        return max(v, float(-g.min())) if len(g) else v

    # ---- exact LP: best radii for given centres ------------------------------------------
    def lp_radii(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = self.n
        w = np.minimum.reduce([x, 1 - x, y, 1 - y])
        w = np.maximum(w, 0.0)
        i, j = self.iu
        d = np.hypot(x[i] - x[j], y[i] - y[j])
        keep = d < w[i] + w[j]  # otherwise implied by r_i<=w_i, r_j<=w_j
        ii, jj, dd = i[keep], j[keep], d[keep]
        m = len(ii)
        if m == 0:
            return w.copy()
        rows = np.repeat(np.arange(m), 2)
        cols = np.stack([ii, jj], 1).ravel()
        A = coo_matrix((np.ones(2 * m), (rows, cols)), shape=(m, n)).tocsr()
        res = linprog(-np.ones(n), A_ub=A, b_ub=dd, bounds=np.stack([np.zeros(n), w], 1), method="highs")
        if res.status != 0:
            return np.zeros(n)
        return np.maximum(res.x, 0.0)

    def lp_value(self, centers: np.ndarray) -> float:
        n = self.n
        return float(self.lp_radii(centers[:n], centers[n:]).sum())

    # ---- SLSQP with analytic Jacobians ---------------------------------------------------
    def _pairs_near(self, z, margin: float):
        x, y, r = self.split(z)
        i, j = self.iu
        d = np.hypot(x[i] - x[j], y[i] - y[j])
        keep = d < 1.5 * (r[i] + r[j]) + margin
        return i[keep], j[keep]

    def _cons(self, I, J):
        n = self.n
        m = len(I)
        lin, linc = self._lin, self._lin_c
        rows = np.arange(m)

        def fun(z):
            x, y, r = z[:n], z[n : 2 * n], z[2 * n :]
            dx = x[I] - x[J]
            dy = y[I] - y[J]
            s = r[I] + r[J]
            return np.concatenate([lin @ z + linc, dx * dx + dy * dy - s * s])

        def jac(z):
            x, y, r = z[:n], z[n : 2 * n], z[2 * n :]
            dx = x[I] - x[J]
            dy = y[I] - y[J]
            s = r[I] + r[J]
            Jp = np.zeros((m, 3 * n))
            Jp[rows, I] = 2 * dx
            Jp[rows, J] = -2 * dx
            Jp[rows, n + I] = 2 * dy
            Jp[rows, n + J] = -2 * dy
            Jp[rows, 2 * n + I] = -2 * s
            Jp[rows, 2 * n + J] = -2 * s
            return np.vstack([lin, Jp])

        return {"type": "ineq", "fun": fun, "jac": jac}

    slp_threshold = 60  # n >= threshold: sparse SLP (search/large.py) instead of dense SLSQP

    def _local_opt_slp(self, z0: np.ndarray) -> LocalResult:
        from .large import slp_local

        z, nit = slp_local(np.array(z0, dtype=float), self.n)
        viol = self.max_violation(z)
        r = self.lp_radii(z[: self.n], z[self.n : 2 * self.n])
        z = np.concatenate([z[: 2 * self.n], r])
        return LocalResult(z, float(r.sum()), viol, nit)

    def local_opt(self, z0: np.ndarray, maxiter: int = 200, margin: float = 0.2) -> LocalResult:
        if self.n >= self.slp_threshold:
            return self._local_opt_slp(z0)
        z = np.array(z0, dtype=float)
        nit = 0
        for _ in range(4):  # add violated far pairs until every pair is satisfied
            I, J = self._pairs_near(z, margin)
            res = minimize(lambda v: float(-np.sum(v[2 * self.n :])), z, jac=lambda v: self._grad,
                           bounds=self.bounds, constraints=[self._cons(I, J)], method="SLSQP",
                           options={"maxiter": maxiter, "ftol": 1e-13})
            z = res.x
            nit += int(res.nit)
            x, y, r = self.split(z)
            i, j = self.iu
            g = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 - (r[i] + r[j]) ** 2
            if g.min() > -1e-7:
                break
            margin *= 2
        viol = self.max_violation(z)
        r = self.lp_radii(z[: self.n], z[self.n : 2 * self.n])
        z = np.concatenate([z[: 2 * self.n], r])
        return LocalResult(z, float(r.sum()), viol, nit)

    # ---- initialisations -----------------------------------------------------------------
    def init_radii(self, x, y) -> np.ndarray:
        return self.lp_radii(x, y)

    def from_centers(self, centers: np.ndarray) -> np.ndarray:
        n = self.n
        c = np.clip(centers, 0.0, 1.0)
        return np.concatenate([c, self.lp_radii(c[:n], c[n:])])

    def random_init(self, rng: np.random.Generator) -> np.ndarray:
        return self.from_centers(rng.random(2 * self.n))

    def hex_init(self, rng: np.random.Generator, jitter: float = 0.02) -> np.ndarray:
        n = self.n
        for _ in range(50):
            rows = int(rng.integers(3, 8))
            cols = int(np.ceil(n / rows)) + int(rng.integers(0, 2))
            pts = []
            for a in range(rows):
                off = 0.5 if a % 2 else 0.0
                for b in range(cols):
                    pts.append(((b + off + 0.5) / (cols + 0.5), (a + 0.5) / rows))
            pts = np.array(pts)
            if len(pts) >= n:
                break
        sel = rng.choice(len(pts), n, replace=False)
        p = pts[sel] + rng.normal(0, jitter, (n, 2))
        if rng.random() < 0.5:
            p = p[:, ::-1]
        return self.from_centers(np.concatenate([p[:, 0], p[:, 1]]))

    def ring_init(self, rng: np.random.Generator, jitter: float = 0.02) -> np.ndarray:
        """Perturbed concentric-ring / corner-grid structures (typical shape of good packings)."""
        n = self.n
        pts: list[tuple[float, float]] = [(0.5, 0.5)]
        k = int(rng.integers(4, 10))
        rad = 0.16
        while len(pts) < n:
            m = max(3, int(round(k * rad / 0.16)))
            ph = rng.random() * 2 * np.pi
            for t in range(m):
                a = ph + 2 * np.pi * t / m
                pts.append((0.5 + rad * np.cos(a) * 1.6, 0.5 + rad * np.sin(a) * 1.6))
            rad += 0.15
        p = np.clip(np.array(pts[:n]) , 0.02, 0.98) + rng.normal(0, jitter, (n, 2))
        return self.from_centers(np.concatenate([p[:, 0], p[:, 1]]))

    def centers_of(self, z) -> np.ndarray:
        return z[: 2 * self.n].copy()

    # ---- certificate export ----------------------------------------------------------------
    def to_certificate(self, z) -> dict:
        x, y, r = self.split(z)
        return {"problem": "sum_radii", "n": self.n,
                "circles": [[repr(float(a)), repr(float(b)), repr(float(c))] for a, b, c in zip(x, y, r)]}


# =============================================================================================
# Problem B: min pairwise distance of n points in the unit square
# =============================================================================================
class MinDistance:
    kind = "min_distance"

    def __init__(self, n: int):
        self.n = n
        self.dim = 2 * n + 1
        self.iu = np.triu_indices(n, 1)
        self.bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 2.0)]
        self._grad = np.zeros(2 * n + 1)
        self._grad[-1] = -1.0

    def split(self, z):
        n = self.n
        return z[:n], z[n : 2 * n], z[-1]

    def dmin(self, z) -> float:
        x, y, _ = self.split(z)
        i, j = self.iu
        return float(np.sqrt(np.min((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)))

    def score_raw(self, z) -> float:
        return self.dmin(z)

    def max_violation(self, z) -> float:
        x, y, _ = self.split(z)
        return float(max(0.0, -x.min(), x.max() - 1, -y.min(), y.max() - 1))

    def _cons(self, I, J):
        n = self.n
        m = len(I)
        rows = np.arange(m)

        def fun(z):
            x, y, t = z[:n], z[n : 2 * n], z[-1]
            return (x[I] - x[J]) ** 2 + (y[I] - y[J]) ** 2 - t

        def jac(z):
            x, y = z[:n], z[n : 2 * n]
            dx = x[I] - x[J]
            dy = y[I] - y[J]
            Jm = np.zeros((m, 2 * n + 1))
            Jm[rows, I] = 2 * dx
            Jm[rows, J] = -2 * dx
            Jm[rows, n + I] = 2 * dy
            Jm[rows, n + J] = -2 * dy
            Jm[:, -1] = -1.0
            return Jm

        return {"type": "ineq", "fun": fun, "jac": jac}

    def local_opt(self, z0: np.ndarray, maxiter: int = 300, factor: float = 2.2) -> LocalResult:
        n = self.n
        z = np.array(z0, dtype=float)
        i, j = self.iu
        nit = 0
        for _ in range(4):
            x, y = z[:n], z[n : 2 * n]
            d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            t = max(z[-1], float(d2.min()))
            keep = d2 < (factor**2) * t + 1e-12
            I, J = i[keep], j[keep]
            res = minimize(lambda v: -v[-1], z, jac=lambda v: self._grad, bounds=self.bounds,
                           constraints=[self._cons(I, J)], method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-15})
            z = res.x
            nit += int(res.nit)
            x, y = z[:n], z[n : 2 * n]
            d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            if d2.min() > z[-1] - 1e-9:
                break
            factor *= 1.5
        z[: 2 * n] = np.clip(z[: 2 * n], 0.0, 1.0)
        z[-1] = self.dmin(z) ** 2
        return LocalResult(z, self.dmin(z), self.max_violation(z), nit)

    def from_points(self, pts: np.ndarray) -> np.ndarray:
        p = np.clip(pts, 0.0, 1.0)
        z = np.concatenate([p, [0.0]])
        z[-1] = self.dmin(z) ** 2
        return z

    def from_centers(self, centers: np.ndarray) -> np.ndarray:
        return self.from_points(centers)

    def random_init(self, rng: np.random.Generator) -> np.ndarray:
        return self.from_points(rng.random(2 * self.n))

    def hex_init(self, rng: np.random.Generator, jitter: float = 0.01) -> np.ndarray:
        n = self.n
        for _ in range(50):
            rows = int(rng.integers(3, 9))
            cols = int(np.ceil(n / rows)) + int(rng.integers(0, 2))
            pts = []
            for a in range(rows):
                off = 0.5 if a % 2 else 0.0
                for b in range(cols):
                    pts.append(((b + off) / (cols - 0.5 + 1e-9), a / max(rows - 1, 1)))
            pts = np.clip(np.array(pts), 0, 1)
            if len(pts) >= n:
                break
        sel = rng.choice(len(pts), n, replace=False)
        p = pts[sel] + rng.normal(0, jitter, (n, 2))
        if rng.random() < 0.5:
            p = p[:, ::-1]
        return self.from_points(np.concatenate([p[:, 0], p[:, 1]]))

    def ring_init(self, rng: np.random.Generator, jitter: float = 0.01) -> np.ndarray:
        return self.hex_init(rng, jitter)

    def centers_of(self, z) -> np.ndarray:
        return z[: 2 * self.n].copy()

    def lp_value(self, centers: np.ndarray) -> float:
        return self.dmin(self.from_points(centers))

    def to_certificate(self, z) -> dict:
        x, y, _ = self.split(z)
        return {"problem": "min_distance", "n": self.n,
                "points": [[repr(float(a)), repr(float(b))] for a, b in zip(x, y)]}


def make_problem(kind: str, n: int):
    if kind == "sum_radii":
        return SumRadii(n)
    if kind == "min_distance":
        return MinDistance(n)
    raise ValueError(kind)
