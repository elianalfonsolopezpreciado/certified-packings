"""Search methods. Every method is a *round function*

    round_fn(problem, rng, iters, state) -> (state, best_z, best_score, n_local_solves)

`state` is a picklable dict (None on the first round), so runs are seedable (rng comes from
(base_seed, worker, round)) and resumable (state is checkpointed between rounds).
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402

from .problems import MinDistance, SumRadii  # noqa: E402


# ------------------------------------------------------------------------------ geometry helpers
def _holes_sum_radii(P: SumRadii, z, skip: list[int], rng, k=4000):
    """Sample positions and return the one with the largest clearance, plus that clearance."""
    n = P.n
    x, y, r = P.split(z)
    keep = np.array([i for i in range(n) if i not in skip], dtype=int)
    pts = rng.random((k, 2))
    cl = np.minimum.reduce([pts[:, 0], 1 - pts[:, 0], pts[:, 1], 1 - pts[:, 1]])
    if len(keep):
        d = np.hypot(pts[:, 0:1] - x[keep][None, :], pts[:, 1:2] - y[keep][None, :]) - r[keep][None, :]
        cl = np.minimum(cl, d.min(1))
    b = int(np.argmax(cl))
    return pts[b], float(cl[b])


def _holes_min_distance(P: MinDistance, z, skip: list[int], rng, k=4000):
    n = P.n
    x, y, _ = P.split(z)
    keep = np.array([i for i in range(n) if i not in skip], dtype=int)
    pts = rng.random((k, 2))
    edge = rng.random((k // 4, 2))  # points on the boundary (optimal point sets touch the walls)
    side = rng.integers(0, 4, k // 4)
    edge[side == 0, 0] = 0.0
    edge[side == 1, 0] = 1.0
    edge[side == 2, 1] = 0.0
    edge[side == 3, 1] = 1.0
    pts = np.concatenate([pts, edge])
    d = np.hypot(pts[:, 0:1] - x[keep][None, :], pts[:, 1:2] - y[keep][None, :]).min(1)
    b = int(np.argmax(d))
    return pts[b], float(d[b])


def _with_point(P, z, i, p, radius=None):
    z = z.copy()
    n = P.n
    z[i] = p[0]
    z[n + i] = p[1]
    if isinstance(P, SumRadii) and radius is not None:
        z[2 * n + i] = max(radius, 0.0)
    return z


# ------------------------------------------------------------------------------ perturbation moves
MOVES_A = ("jitter_small", "jitter_big", "reinsert_smallest", "reinsert_k", "swap_radii", "relocate_one")
MOVES_A_LARGE = ("jitter_small", "cluster_jitter", "reinsert_smallest", "reinsert_k", "swap_radii", "relocate_one",
                 "cluster_jitter", "reinsert_smallest")  # n >= LARGE_N: local moves, scaled by the mean radius
LARGE_N = 60
MOVES_B = ("jitter_small", "jitter_big", "reinsert_worst", "reinsert_k", "relocate_one")


def perturb(P, z: np.ndarray, rng: np.random.Generator, move: str | None = None) -> tuple[np.ndarray, str]:
    n = P.n
    is_a = isinstance(P, SumRadii)
    large = is_a and n >= LARGE_N
    if move is None:
        move = str(rng.choice((MOVES_A_LARGE if large else MOVES_A) if is_a else MOVES_B))
    holes = _holes_sum_radii if is_a else _holes_min_distance
    z = z.copy()
    if large and move in ("jitter_small", "jitter_big"):
        rbar = float(z[2 * n :].mean())
        s = (0.05 if move == "jitter_small" else 0.25) * rbar
        c = np.clip(z[: 2 * n] + rng.normal(0, s * (1 + rng.random()), 2 * n), 0, 1)
        return P.from_centers(c), move
    if move == "cluster_jitter":  # jitter only the circles inside a random disc (a few diameters wide)
        x, y, r = P.split(z) if is_a else (z[:n], z[n : 2 * n], None)
        rbar = float(r.mean()) if r is not None else 0.05
        cx, cy = rng.random(2)
        rad = rbar * float(rng.uniform(3.0, 7.0))
        m = np.hypot(x - cx, y - cy) < rad
        if m.sum() < 2:
            m[np.argsort(np.hypot(x - cx, y - cy))[:3]] = True
        s = rbar * float(rng.choice([0.15, 0.3, 0.6]))
        c = np.concatenate([x, y]).copy()
        c[:n][m] += rng.normal(0, s, int(m.sum()))
        c[n:][m] += rng.normal(0, s, int(m.sum()))
        c = np.clip(c, 0, 1)
        return (P.from_centers(c) if is_a else P.from_points(c)), move
    if move in ("jitter_small", "jitter_big"):
        s = 0.004 if move == "jitter_small" else 0.03
        c = np.clip(z[: 2 * n] + rng.normal(0, s * (1 + rng.random()), 2 * n), 0, 1)
        return (P.from_centers(c) if is_a else P.from_points(c)), move
    if large and move == "reinsert_smallest":  # the k smallest circles jump into the largest voids
        k = int(rng.integers(1, 4))
        idx = [int(t) for t in np.argsort(z[2 * n :])[:k]]
        placed: list[int] = []
        for i in idx:
            p, cl = holes(P, z, [t for t in idx if t not in placed], rng)
            z = _with_point(P, z, i, p, cl)
            placed.append(i)
        return P.from_centers(z[: 2 * n]), move
    if move in ("reinsert_smallest", "reinsert_worst"):
        if is_a:
            i = int(np.argmin(z[2 * n :]))
        else:  # a point of the closest pair
            x, y, _ = P.split(z)
            ii, jj = P.iu
            d = np.hypot(x[ii] - x[jj], y[ii] - y[jj])
            k = int(np.argmin(d))
            i = int(ii[k] if rng.random() < 0.5 else jj[k])
        p, cl = holes(P, z, [i], rng)
        z = _with_point(P, z, i, p, cl)
        c = z[: 2 * n]
        return (P.from_centers(c) if is_a else P.from_points(c)), move
    if move == "reinsert_k":
        k = int(rng.integers(2, 5))
        idx = [int(t) for t in rng.choice(n, min(k, n), replace=False)]
        placed: list[int] = []  # place one by one at the biggest holes (not-yet-placed ones are ignored)
        for i in idx:
            p, cl = holes(P, z, [t for t in idx if t not in placed], rng)
            z = _with_point(P, z, i, p, cl)
            placed.append(i)
        c = z[: 2 * n]
        return (P.from_centers(c) if is_a else P.from_points(c)), move
    if move == "relocate_one":
        i = int(rng.integers(n))
        p = rng.random(2)
        z = _with_point(P, z, i, p, 0.0)
        c = z[: 2 * n]
        return (P.from_centers(c) if is_a else P.from_points(c)), move
    if move == "swap_radii" and is_a:
        x, y, r = P.split(z)
        i = int(rng.integers(n))
        d = np.hypot(x - x[i], y - y[i])
        d[i] = 9
        j = int(np.argsort(d)[int(rng.integers(0, min(4, n - 1)))])  # a spatial neighbour
        z[2 * n + i], z[2 * n + j] = z[2 * n + j], z[2 * n + i]
        return z, move
    return perturb(P, z, rng, "jitter_small")


def convert_to_n(P, z_other: np.ndarray, n_other: int, rng) -> np.ndarray | None:
    """Turn a sum-of-radii solution for n_other circles into a start for P.n circles: n_other = n-1 -> insert a circle
    into the largest void, n_other = n+1 -> delete the smallest circle, n_other = n -> unchanged. Others -> None."""
    n = P.n
    if not isinstance(P, SumRadii):
        return None
    x, y, r = z_other[:n_other], z_other[n_other : 2 * n_other], z_other[2 * n_other :]
    if n_other == n:
        return z_other.copy()
    if n_other == n + 1:
        keep = np.argsort(r)[1:]
        return P.from_centers(np.concatenate([x[keep], y[keep]]))
    if n_other == n - 1:
        tmp = np.concatenate([x, [0.5], y, [0.5], r, [0.0]])
        Pn = SumRadii(n_other + 1)
        p, _ = _holes_sum_radii(Pn, tmp, [n_other], rng)
        return P.from_centers(np.concatenate([np.append(x, p[0]), np.append(y, p[1])]))
    return None


def start_point(P, rng) -> np.ndarray:
    """Initial point for a chain: a seed solution (if the driver provided some, 60% of the time, lightly perturbed) or a
    fresh structured start."""
    seeds = getattr(P, "seed_zs", None)
    if seeds and rng.random() < 0.6:
        z = seeds[int(rng.integers(len(seeds)))]
        if rng.random() < 0.5:
            z, _ = perturb(P, z, rng)
        return z
    return mixed_init(P, rng)


def mixed_init(P, rng) -> np.ndarray:
    u = rng.random()
    if u < 0.4:
        return P.random_init(rng)
    if u < 0.75:
        return P.hex_init(rng)
    return P.ring_init(rng)


# ------------------------------------------------------------------------------ 1. multistart
def multistart_round(P, rng, iters, state):
    best_z, best_s = (state["best_z"], state["best_s"]) if state else (None, -np.inf)
    for _ in range(iters):
        res = P.local_opt(mixed_init(P, rng))
        if res.score > best_s:
            best_z, best_s = res.z, res.score
    return {"best_z": best_z, "best_s": best_s}, best_z, best_s, iters


# ------------------------------------------------------------------------------ 2. basin hopping
def basin_hopping_round(P, rng, iters, state, temp0=0.004, temp1=0.0005, patience=120):
    if state is None:
        res = P.local_opt(start_point(P, rng))
        state = {"z": res.z, "s": res.score, "best_z": res.z, "best_s": res.score, "stale": 0, "it": 0,
                 "moves": {}}
        solves = 1
    else:
        solves = 0
    z, s = state["z"], state["s"]
    for _ in range(iters):
        state["it"] += 1
        frac = (state["it"] % 400) / 400.0  # cyclic annealing
        T = temp0 * (temp1 / temp0) ** frac
        z2, mv = perturb(P, z, rng)
        res = P.local_opt(z2)
        solves += 1
        st = state["moves"].setdefault(mv, [0, 0])
        st[0] += 1
        if res.score > s + 1e-12:
            st[1] += 1
        if res.score >= s or rng.random() < np.exp((res.score - s) / T):
            z, s = res.z, res.score
        if s > state["best_s"] + 1e-12:
            state["best_z"], state["best_s"], state["stale"] = z, s, 0
        else:
            state["stale"] += 1
        if state["stale"] > patience:  # restart from a fresh structure, keep the incumbent best
            res = P.local_opt(start_point(P, rng))
            solves += 1
            z, s = res.z, res.score
            state["stale"] = 0
            if s > state["best_s"]:
                state["best_z"], state["best_s"] = z, s
    state["z"], state["s"] = z, s
    return state, state["best_z"], state["best_s"], solves


# ------------------------------------------------------------------------------ 3. CMA-ES over centres
def cmaes_round(P, rng, iters, state, popsize=None, sigma0=0.12, elites=6):
    """One CMA-ES run over centres (fitness = exact LP value / min distance), then SLSQP polishing.

    `iters` = number of fitness evaluations."""
    import cma

    n = P.n
    x0 = P.centers_of(mixed_init(P, rng))
    opts = {"bounds": [0.0, 1.0], "seed": int(rng.integers(1, 2**31 - 1)), "verbose": -9,
            "popsize": popsize or (16 + 2 * n // 3), "tolfun": 1e-10}
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    pool: list[tuple[float, np.ndarray]] = []
    evals = 0
    while evals < iters and not es.stop():
        X = es.ask()
        F = [-P.lp_value(np.asarray(x)) for x in X]
        evals += len(X)
        es.tell(X, F)
        for f, x in zip(F, X):
            pool.append((f, np.asarray(x)))
        if len(pool) > 200:
            pool.sort(key=lambda t: t[0])
            pool = pool[:60]
    pool.sort(key=lambda t: t[0])
    best_z, best_s = (state["best_z"], state["best_s"]) if state else (None, -np.inf)
    seen: list[float] = []
    solves = 0
    for f, x in pool:
        if len(seen) >= elites:
            break
        if any(abs(f - g) < 1e-7 for g in seen):
            continue
        seen.append(f)
        res = P.local_opt(P.from_centers(x))
        solves += 1
        if res.score > best_s:
            best_z, best_s = res.z, res.score
    return {"best_z": best_z, "best_s": best_s}, best_z, best_s, solves


# ------------------------------------------------------------------------------ 4. differential evolution
def de_round(P, rng, iters, state, elites=6):
    """scipy differential_evolution over centres; `iters` = generations."""
    from scipy.optimize import differential_evolution

    n = P.n
    res = differential_evolution(lambda c: -P.lp_value(c), [(0.0, 1.0)] * (2 * n), maxiter=iters, popsize=8,
                                 seed=int(rng.integers(1, 2**31 - 1)), polish=False, tol=0, atol=0,
                                 mutation=(0.4, 0.9), recombination=0.8, init="sobol", updating="immediate")
    order = np.argsort(res.population_energies)
    best_z, best_s = (state["best_z"], state["best_s"]) if state else (None, -np.inf)
    seen: list[float] = []
    solves = 0
    for k in order:
        f = float(res.population_energies[k])
        if len(seen) >= elites:
            break
        if any(abs(f - g) < 1e-7 for g in seen):
            continue
        seen.append(f)
        r2 = P.local_opt(P.from_centers(res.population[k]))
        solves += 1
        if r2.score > best_s:
            best_z, best_s = r2.z, r2.score
    return {"best_z": best_z, "best_s": best_s}, best_z, best_s, solves


# ------------------------------------------------------------------------------ 5. evolutionary loop
def _signature(P, z) -> np.ndarray:
    if isinstance(P, SumRadii):
        return np.sort(z[2 * P.n :])
    return np.array([P.dmin(z)])


def _sig_dist(P, a, b) -> float:
    return float(np.max(np.abs(a - b)))


def crossover(P, za, zb, rng) -> np.ndarray:
    """Cut both parents by a random line; take one side from each, then repair the count to n."""
    n = P.n
    ang = rng.random() * np.pi
    nx, ny = np.cos(ang), np.sin(ang)
    off = 0.5 + 0.2 * (rng.random() - 0.5)
    pa = np.stack([za[:n], za[n : 2 * n]], 1)
    pb = np.stack([zb[:n], zb[n : 2 * n]], 1)
    ma = (pa[:, 0] - 0.5) * nx + (pa[:, 1] - 0.5) * ny + (0.5 - off) > 0
    mb = ~((pb[:, 0] - 0.5) * nx + (pb[:, 1] - 0.5) * ny + (0.5 - off) > 0)
    pts = np.concatenate([pa[ma], pb[mb]])
    if len(pts) > n:
        # drop the points involved in the tightest contacts first
        while len(pts) > n:
            d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
            np.fill_diagonal(d, 9)
            i = int(np.argmin(d.min(1)))
            pts = np.delete(pts, i, 0)
    holes = _holes_sum_radii if isinstance(P, SumRadii) else _holes_min_distance
    while len(pts) < n:
        tmp = np.zeros(P.dim)
        m = len(pts)
        if isinstance(P, SumRadii):
            tmp = np.concatenate([pts[:, 0], np.zeros(n - m), pts[:, 1], np.zeros(n - m), np.full(n, 0.0)])
            rr = P.lp_radii(np.concatenate([pts[:, 0], np.full(n - m, 0.5)]),
                            np.concatenate([pts[:, 1], np.full(n - m, 0.5)]))[:m]
            tmp = np.concatenate([np.concatenate([pts[:, 0], np.full(n - m, 0.5)]),
                                  np.concatenate([pts[:, 1], np.full(n - m, 0.5)]),
                                  np.concatenate([rr, np.zeros(n - m)])])
        else:
            tmp = np.concatenate([np.concatenate([pts[:, 0], np.full(n - m, 0.5)]),
                                  np.concatenate([pts[:, 1], np.full(n - m, 0.5)]), [0.0]])
        p, _ = holes(P, tmp, list(range(m, n)), rng)
        pts = np.concatenate([pts, p[None, :]])
    c = np.concatenate([pts[:, 0], pts[:, 1]])
    return P.from_centers(c)


def _insert(P, pop: list[dict], z, s, pop_size: int, delta: float) -> bool:
    sig = _signature(P, z)
    for m in pop:
        if _sig_dist(P, sig, m["sig"]) < delta:  # same niche: replace only if strictly better
            if s > m["s"] + 1e-12:
                m.update(z=z, s=s, sig=sig)
                return True
            return False
    if len(pop) < pop_size:
        pop.append({"z": z, "s": s, "sig": sig})
        return True
    w = min(range(len(pop)), key=lambda k: pop[k]["s"])
    if s > pop[w]["s"]:
        pop[w] = {"z": z, "s": s, "sig": sig}
        return True
    return False


def evo_round(P, rng, iters, state, pop_size=24, p_cross=0.3):
    delta = 3e-4 if isinstance(P, SumRadii) else 1e-9
    solves = 0
    if state is None:
        state = {"pop": [], "gen": 0}
    pop = state["pop"]
    while len(pop) < pop_size:  # initial population
        res = P.local_opt(start_point(P, rng))
        solves += 1
        _insert(P, pop, res.z, res.score, pop_size, delta)
        if solves > 4 * pop_size:
            break
    for _ in range(iters):
        state["gen"] += 1
        # tournament selection
        def pick():
            a, b = rng.choice(len(pop), 2, replace=len(pop) < 2)
            return pop[a] if pop[a]["s"] >= pop[b]["s"] else pop[b]

        if len(pop) >= 2 and rng.random() < p_cross:
            pa, pb = pick(), pick()
            start = crossover(P, pa["z"], pb["z"], rng)
            if rng.random() < 0.5:
                start, _ = perturb(P, start, rng, "jitter_small")
        else:
            start, _ = perturb(P, pick()["z"], rng)
        res = P.local_opt(start)
        solves += 1
        _insert(P, pop, res.z, res.score, pop_size, delta)
    best = max(pop, key=lambda m: m["s"])
    return state, best["z"], best["s"], solves


def migrate(states: list[dict], k: int = 2) -> list[dict]:
    """Island migration for evo: each island receives the k best members of the next island."""
    if len(states) < 2 or "pop" not in (states[0] or {}):
        return states
    tops = [sorted(s["pop"], key=lambda m: -m["s"])[:k] for s in states]
    for i, s in enumerate(states):
        for m in tops[(i + 1) % len(states)]:
            if all(np.max(np.abs(m["sig"] - q["sig"])) > 1e-9 for q in s["pop"]) and len(s["pop"]) > 0:
                w = min(range(len(s["pop"])), key=lambda t: s["pop"][t]["s"])
                if m["s"] > s["pop"][w]["s"]:
                    s["pop"][w] = dict(m)
    return states


METHODS = {"multistart": multistart_round, "basin_hopping": basin_hopping_round, "cmaes": cmaes_round,
           "de": de_round, "evo": evo_round}
