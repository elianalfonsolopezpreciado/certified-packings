"""Problem C: first autocorrelation inequality (upper bound on C1), step-function search.

Minimise   R(a) = 2N * max_m c_m / (sum a)^2,   c_m = sum_{i+j=m} a_i a_j,  a_i >= 0  (N pieces
on [-1/4, 1/4]).   (See verify/core.py for why this equals max(f*f)/(int f)^2.)

Stage 1 (GPU): B random restarts, a = exp(theta), p-norm smoothing of max_m c_m with p annealed, Adam.
Stage 2 (CPU): sequential LP with trust region and an active-row cutting-plane LP (HiGHS):
    min t  s.t. c_m + 2 (a * d)_m <= t on near-active rows, sum d = 0, -a <= d <= Delta.
Only floats here; certification is by `python -m verify` on certificates/autocorr1_N<k>.json.

    python -m search.autocorr --N 95 --restarts 2048 --seconds 120
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
from scipy.linalg import toeplitz  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

RECORDS = {  # upper bounds on C1 as quoted in arXiv:2601.16175 Table 2 (retrieved 2026-09-28)
    "best_human_51": 1.50973, "alphaevolve_95": 1.50530, "alphaevolve_v2_1319": 1.50317,
    "thetaevolve_1319": 1.50314, "ttt_discover_30000": 1.50287}


def ratio(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    return float(2 * len(a) * np.convolve(a, a).max() / a.sum() ** 2)


# ------------------------------------------------------------------------------ stage 1: GPU
def gpu_descent(N: int, B: int, steps: int, seed: int, top: int = 8):
    import torch

    dev = torch.device("cuda")
    dt = torch.float64
    g = torch.Generator(device=dev).manual_seed(seed)
    theta = (0.7 * torch.randn(B, N, generator=g, device=dev, dtype=dt)).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=0.05)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        p = 8.0 * (600.0 / 8.0) ** f
        for gr in opt.param_groups:
            gr["lr"] = 0.05 * (0.02 ** f)
        opt.zero_grad(set_to_none=True)
        a = torch.exp(theta)
        fa = torch.fft.rfft(a, n=2 * N)
        c = torch.fft.irfft(fa * fa, n=2 * N)[:, : 2 * N - 1].clamp_min(1e-30)
        cmax = c.amax(1, keepdim=True).detach()
        lse = cmax.squeeze(1) * ((c / cmax) ** p).sum(1) ** (1.0 / p)  # smooth max (p-norm)
        loss = (torch.log(2 * N * lse) - 2 * torch.log(a.sum(1))).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        a = torch.exp(theta)
        fa = torch.fft.rfft(a, n=2 * N)
        c = torch.fft.irfft(fa * fa, n=2 * N)[:, : 2 * N - 1]
        R = 2 * N * c.amax(1) / a.sum(1) ** 2
        idx = torch.argsort(R)[:top]
        return a[idx].cpu().numpy(), R[idx].cpu().numpy(), float(R.median())


# ------------------------------------------------------------------------------ stage 2: SLP
def slp(a0: np.ndarray, seconds: float, delta0: float = 0.3, verbose: bool = False):
    a = np.asarray(a0, dtype=float).copy()
    N = len(a)
    a *= N / a.sum()  # normalise sum = N
    best = ratio(a)
    delta = delta0
    t0 = time.time()
    it = 0
    hist = []
    while time.time() - t0 < seconds and delta > 1e-9:
        it += 1
        c = np.convolve(a, a)
        cmax = c.max()
        rows = np.where(c >= cmax - max(0.15 * cmax, 1e-9))[0]  # near-active rows
        # (a*d)_m = sum_i a_{m-i} d_i  -> Toeplitz matrix of a restricted to rows
        full = toeplitz(np.concatenate([a, np.zeros(N - 1)]), np.concatenate([[a[0]], np.zeros(N - 1)]))  # (2N-1)xN
        A = 2.0 * full[rows]
        # variables: d (N), t ; minimise t ; A d - t <= -c_rows
        Aub = np.hstack([A, -np.ones((len(rows), 1))])
        bub = -c[rows]
        Aeq = np.concatenate([np.ones(N), [0.0]])[None, :]
        bounds = [(max(-a[i], -delta), delta) for i in range(N)] + [(None, None)]  # a has mean 1
        cost = np.zeros(N + 1)
        cost[-1] = 1.0
        res = linprog(cost, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=[0.0], bounds=bounds, method="highs")
        if res.status != 0:
            delta *= 0.5
            continue
        d = res.x[:N]
        cand = np.maximum(a + d, 0.0)
        r = ratio(cand)
        if r < best - 1e-13:
            a, best = cand * (N / cand.sum()), r
            delta = min(delta * 1.5, 1.0)
        else:
            delta *= 0.5
        hist.append((time.time() - t0, best))
        if verbose and it % 10 == 0:
            print(f"  slp it{it} R={best:.8f} delta={delta:.2e} rows={len(rows)}", flush=True)
    return a, best, hist


def _bh_chain(args):
    """Basin hopping in step-function space: perturb -> SLP -> accept if better (small Metropolis)."""
    a0, seed, iters, slp_s = args
    rng = np.random.default_rng(seed)
    a, r = np.asarray(a0, float).copy(), ratio(a0)
    best_a, best_r = a.copy(), r
    N = len(a)
    for _ in range(iters):
        b = a.copy()
        u = rng.random()
        if u < 0.4:
            b *= np.exp(rng.normal(0, rng.choice([0.02, 0.05, 0.15]), N))
        elif u < 0.7:  # smooth low-frequency deformation
            k = rng.integers(1, 6)
            x = np.linspace(0, 1, N)
            b *= np.exp(rng.normal(0, 0.1) * np.cos(np.pi * k * x + rng.random() * 6.28))
        else:  # re-randomise a block
            L = max(2, N // rng.integers(4, 12))
            i0 = rng.integers(0, N - L + 1)
            b[i0 : i0 + L] = rng.random(L) * b.mean() * 2
        b, rb, _ = slp(np.maximum(b, 1e-6), slp_s)
        if rb < r or rng.random() < np.exp(-(rb - r) / 2e-4):
            a, r = b, rb
        if r < best_r:
            best_a, best_r = a.copy(), r
    return best_a, best_r


def bh(N: int, seed: int, seconds: float, workers: int, slp_s: float = 4.0, start=None, iters: int = 6):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    if start is None:
        cand, R, _ = gpu_descent(N, 2048, 3000, seed, top=workers)
        starts = [slp(cand[k], 20)[0] for k in range(workers)]
    else:
        starts = [np.asarray(start, float)] * workers
    best_a, best_r = None, 1e9
    t0 = time.time()
    rnd = 0
    hist = []
    with ctx.Pool(workers) as pool:
        while time.time() - t0 < seconds:
            outs = pool.map(_bh_chain, [(starts[w], seed * 1000 + rnd * workers + w, iters, slp_s) for w in range(workers)])
            rnd += 1
            for w, (a, r) in enumerate(outs):
                starts[w] = a
                if r < best_r:
                    best_a, best_r = a.copy(), r
            # elitist restart: worst chain restarts from the best
            worst = int(np.argmax([ratio(x) for x in starts]))
            starts[worst] = best_a.copy()
            hist.append((time.time() - t0, best_r))
            print(f"[bh N={N}] round {rnd} t={time.time()-t0:.0f}s best R={best_r:.8f}", flush=True)
    return best_a, best_r, hist


def certificate(a: np.ndarray) -> dict:
    return {"problem": "autocorr1", "n": len(a), "heights": [repr(float(v)) for v in a]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=95)
    ap.add_argument("--restarts", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seconds", type=float, default=120, help="SLP seconds per candidate")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bh-seconds", type=float, default=0, help="if >0: basin hopping (SLP) for this long")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    if a.bh_seconds > 0:
        out_dir = ROOT / "results" / "autocorr1"
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"best_N{a.N}.json"
        start = np.array([float(v) for v in json.load(open(p))["heights"]]) if p.exists() else None
        best_a, best_r, hist = bh(a.N, a.seed, a.bh_seconds, a.workers, start=start)
        old = json.load(open(p))["ratio_float"] if p.exists() else 1e9
        if best_r < old:
            json.dump(certificate(best_a) | {"ratio_float": best_r, "seed": a.seed}, open(p, "w"), indent=1)
        json.dump({"N": a.N, "mode": "bh", "seed": a.seed, "seconds": a.bh_seconds, "workers": a.workers,
                   "best": best_r, "hist": hist}, open(out_dir / f"bh_N{a.N}_seed{a.seed}.json", "w"), indent=1)
        print(f"[N={a.N}] BH best float R = {best_r:.9f}")
        return
    cand, R, med = gpu_descent(a.N, a.restarts, a.steps, a.seed, top=a.top)
    print(f"[N={a.N}] GPU stage {time.time()-t0:.0f}s best R={R.min():.6f} median={med:.4f}", flush=True)
    out_dir = ROOT / "results" / "autocorr1"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_a, best_r = None, 1e9
    log = []
    for k in range(len(cand)):
        aa, rr, hist = slp(cand[k], a.seconds)
        print(f"[N={a.N}] cand {k}: GPU R={R[k]:.6f} -> SLP R={rr:.8f}", flush=True)
        log.append({"cand": k, "gpu_R": float(R[k]), "slp_R": rr})
        if rr < best_r:
            best_a, best_r = aa, rr
    p = out_dir / f"best_N{a.N}.json"
    old = json.load(open(p))["ratio_float"] if p.exists() else 1e9
    if best_r < old:
        json.dump(certificate(best_a) | {"ratio_float": best_r, "seed": a.seed}, open(p, "w"), indent=1)
    json.dump({"N": a.N, "seed": a.seed, "restarts": a.restarts, "steps": a.steps, "log": log,
               "best": best_r, "gpu_median": med, "elapsed": time.time() - t0},
              open(out_dir / f"run_N{a.N}_seed{a.seed}.json", "w"), indent=1)
    print(f"[N={a.N}] best float R = {best_r:.9f}  (records: {RECORDS})", flush=True)


if __name__ == "__main__":
    main()
