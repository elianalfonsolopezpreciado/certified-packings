"""GPU stage (Problem A): batched penalty-formulation search on CUDA, CPU polishing of the top-k.

Modes
  multistart : B random / hex-lattice starts, annealed penalty weight, Adam.
  bh         : B parallel basin-hopping chains (perturb -> short Adam re-optimisation -> accept).

GPU numbers are only *screening* scores. The top-k configurations are re-optimised on CPU with
SLSQP (search.problems.SumRadii.local_opt) and polished/verified by the CPU pipeline. GPU results
count for nothing until the independent verifier confirms a polished certificate.

    python -m search.gpu configs/gpu_a26.yaml
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _torch():
    import torch

    return torch


# ------------------------------------------------------------------------------ core tensors
def pair_violation(x, r, mask):
    """x [B,n,2], r [B,n] -> viol [B,n,n] = relu(r_i + r_j - d_ij) on the strict upper triangle."""
    torch = _torch()
    dxy = x[:, :, None, :] - x[:, None, :, :]
    d = torch.sqrt((dxy * dxy).sum(-1) + 1e-14)
    return torch.relu(r[:, :, None] + r[:, None, :] - d) * mask


def wall_violation(x, r):
    torch = _torch()
    return (torch.relu(r - x[..., 0]), torch.relu(x[..., 0] + r - 1), torch.relu(r - x[..., 1]),
            torch.relu(x[..., 1] + r - 1))


def penalty(x, r, mask):
    torch = _torch()
    vp = pair_violation(x, r, mask)
    w = wall_violation(x, r)
    return (vp * vp).sum((1, 2)) + sum((t * t).sum(1) for t in w) + (torch.relu(-r) ** 2).sum(1)


def feasible_score(x, r, mask):
    """Conservative feasible sum of radii: shrink each radius by its largest violation (see docstring
    of the derivation in the report: r'_i = r_i - max_j viol_ij - wall_i is always feasible)."""
    torch = _torch()
    vp = pair_violation(x, r, mask)
    vpi = torch.maximum(vp.amax(2), vp.amax(1))
    w = wall_violation(x, r)
    vw = torch.maximum(torch.maximum(w[0], w[1]), torch.maximum(w[2], w[3]))
    rr = torch.clamp(r - vpi - vw, min=0.0)
    return rr.sum(1)


def adam_optimize(x, r, mask, steps, mu0, mu1, lr0, lr1):
    torch = _torch()
    x = x.clone().requires_grad_(True)
    r = r.clone().requires_grad_(True)
    opt = torch.optim.Adam([x, r], lr=lr0)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        mu = mu0 * (mu1 / mu0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss = (-r.sum(1) + mu * penalty(x, r, mask)).sum()
        loss.backward()
        opt.step()
    return x.detach(), r.detach()


# ------------------------------------------------------------------------------ initialisations
def init_batch(B, n, gen, device, dtype, frac_hex=0.5, r0=0.02):
    torch = _torch()
    x = torch.rand(B, n, 2, generator=gen, device=device, dtype=dtype)
    nh = int(B * frac_hex)
    if nh:
        groups = [(rows, cols) for rows in range(3, 8) for cols in (int(np.ceil(n / rows)), int(np.ceil(n / rows)) + 1)]
        per = max(nh // len(groups), 1)
        pos = 0
        for rows, cols in groups:
            if pos >= nh:
                break
            cnt = min(per, nh - pos)
            pts = []
            for a in range(rows):
                off = 0.5 if a % 2 else 0.0
                for b in range(cols):
                    pts.append(((b + off + 0.5) / (cols + 0.5), (a + 0.5) / rows))
            pts_t = torch.tensor(pts, device=device, dtype=dtype)
            if len(pts_t) < n:
                continue
            perm = torch.rand(cnt, len(pts_t), generator=gen, device=device, dtype=dtype).argsort(1)[:, :n]
            sel = pts_t[perm] + 0.02 * torch.randn(cnt, n, 2, generator=gen, device=device, dtype=dtype)
            if rows % 2 == 0:
                sel = sel.flip(-1)
            x[pos : pos + cnt] = sel
            pos += cnt
    x = x.clamp(0.02, 0.98)
    r = torch.full((B, n), r0, device=device, dtype=dtype)
    return x, r


def perturb_batch(x, r, gen, mask):
    """Random move per chain: 0 jitter-small, 1 jitter-big, 2 reinsert smallest at best of K samples,
    3 relocate a random circle to a random spot, 4 swap radii of two neighbours."""
    torch = _torch()
    B, n, _ = x.shape
    dev, dt = x.device, x.dtype
    move = torch.randint(0, 5, (B,), generator=gen, device=dev)
    x2, r2 = x.clone(), r.clone()
    ar = torch.arange(B, device=dev)
    # jitter
    js = (move == 0)[:, None, None] * 0.004 + (move == 1)[:, None, None] * 0.03
    x2 = x2 + js * (1 + torch.rand(B, 1, 1, generator=gen, device=dev, dtype=dt)) * torch.randn(
        B, n, 2, generator=gen, device=dev, dtype=dt)
    # reinsert smallest at best hole (K samples)
    K = 96
    cand = torch.rand(B, K, 2, generator=gen, device=dev, dtype=dt)
    idx_small = r.argmin(1)
    other = torch.ones(B, n, dtype=torch.bool, device=dev)
    other[ar, idx_small] = False
    dist = torch.sqrt(((cand[:, :, None, :] - x[:, None, :, :]) ** 2).sum(-1) + 1e-14) - r[:, None, :]
    dist = torch.where(other[:, None, :], dist, torch.full_like(dist, 9.0))
    wallc = torch.minimum(torch.minimum(cand[..., 0], 1 - cand[..., 0]), torch.minimum(cand[..., 1], 1 - cand[..., 1]))
    cl = torch.minimum(dist.amin(2), wallc)
    best = cl.argmax(1)
    p = cand[ar, best]
    m2 = move == 2
    x2[ar[m2], idx_small[m2]] = p[m2]
    r2[ar[m2], idx_small[m2]] = torch.clamp(cl[ar, best][m2], min=0.0)
    # relocate random circle
    m3 = move == 3
    i3 = torch.randint(0, n, (B,), generator=gen, device=dev)
    x2[ar[m3], i3[m3]] = torch.rand(int(m3.sum()), 2, generator=gen, device=dev, dtype=dt)
    r2[ar[m3], i3[m3]] = 0.0
    # swap radii of a circle and its 2nd nearest neighbour
    m4 = move == 4
    i4 = torch.randint(0, n, (B,), generator=gen, device=dev)
    d = torch.sqrt(((x - x[ar, i4][:, None, :]) ** 2).sum(-1))
    d[ar, i4] = 9.0
    j4 = d.argsort(1)[:, torch.randint(0, 3, (1,), generator=gen, device=dev).item()]
    ri, rj = r[ar, i4].clone(), r[ar, j4].clone()
    r2[ar[m4], i4[m4]] = rj[m4]
    r2[ar[m4], j4[m4]] = ri[m4]
    return x2.clamp(0.0, 1.0), r2, move


# ------------------------------------------------------------------------------ CPU polishing pool
def _polish_one(args):
    from search.problems import SumRadii

    n, z = args
    P = SumRadii(n)
    res = P.local_opt(z, maxiter=300)
    return res.score, res.z


def polish_topk(pool, n, x, r, scores, k, seen_scores):
    """SLSQP-polish the k best distinct GPU candidates on CPU. Returns list of (score, z)."""
    order = np.argsort(-scores)
    picked = []
    last = []
    for i in order:
        s = float(scores[i])
        if any(abs(s - t) < 2e-6 for t in last):
            continue
        last.append(s)
        z = np.concatenate([x[i, :, 0], x[i, :, 1], np.maximum(r[i], 0.0)])
        picked.append((n, z))
        if len(picked) >= k:
            break
    return pool.map(_polish_one, picked, chunksize=1)


# ------------------------------------------------------------------------------ large-n stage
def init_structured(B, n, gen, dev, dtype, mix=(0.3, 0.4, 0.15, 0.15)):
    """Large-n starts: uniform random | hex lattice with vacancies | hex + radius grading | hex + strong jitter."""
    torch = _torch()
    x = torch.rand(B, n, 2, generator=gen, device=dev, dtype=dtype)
    r = torch.full((B, n), 0.4 / np.sqrt(n), device=dev, dtype=dtype)
    kinds = torch.multinomial(torch.tensor(mix, dtype=torch.float64), B, replacement=True, generator=torch.Generator().manual_seed(
        int(torch.randint(0, 2**31 - 1, (1,), generator=gen, device=dev).item())))
    kinds = kinds.to(dev)
    k0, k1 = max(int(np.sqrt(n) * 0.8), 3), int(np.sqrt(n) * 1.3) + 1
    configs = [(rows, cols) for rows in range(k0, k1 + 1) for cols in (int(np.ceil(n / rows)), int(np.ceil(n / rows)) + 1)]
    lat = {}
    for rows, cols in configs:
        pts = []
        for a in range(rows):
            off = 0.5 if a % 2 else 0.0
            for b in range(cols):
                pts.append(((b + off + 0.5) / (cols + 0.5), (a + 0.5) / rows))
        if len(pts) >= n:
            lat[(rows, cols)] = torch.tensor(pts, device=dev, dtype=dtype)
    keys = list(lat)
    for kind in (1, 2, 3):
        idx = (kinds == kind).nonzero().squeeze(1)
        if len(idx) == 0 or not keys:
            continue
        pick = torch.randint(0, len(keys), (len(idx),), generator=gen, device=dev)
        for ki, key in enumerate(keys):
            sel = idx[pick == ki]
            if len(sel) == 0:
                continue
            L = lat[key]
            perm = torch.rand(len(sel), len(L), generator=gen, device=dev, dtype=dtype).argsort(1)[:, :n]
            jit = (0.35 if kind == 3 else 0.06) / np.sqrt(n)
            pts = L[perm] + jit * torch.randn(len(sel), n, 2, generator=gen, device=dev, dtype=dtype)
            if torch.rand(1, generator=gen, device=dev).item() < 0.5:
                pts = pts.flip(-1)
            x[sel] = pts
            if kind == 2:  # graded radii: larger in the interior
                wall = torch.minimum(torch.minimum(pts[..., 0], 1 - pts[..., 0]), torch.minimum(pts[..., 1], 1 - pts[..., 1]))
                r[sel] = (0.4 / np.sqrt(n)) * (0.6 + 1.2 * wall.clamp(0, 0.5) / 0.5) * (0.8 + 0.4 * torch.rand(
                    len(sel), n, generator=gen, device=dev, dtype=dtype))
    return x.clamp(0.01, 0.99), r


def run_large(cfg: dict) -> dict:
    torch = _torch()
    from search.driver import classify, load_record, update_leaderboard
    from search.problems import SumRadii

    n = int(cfg["n"])
    name = cfg.get("name", f"gpu_large_sum_radii_n{n}")
    out_dir = ROOT / "results" / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "log.jsonl", "a")
    seed = int(cfg.get("seed", 0))
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)
    tlimit = float(cfg.get("time_limit_s", 600))
    # memory budget: ~56*n^2 bytes per config (float32 temporaries), x2 for the float64 refinement
    Bmax = int(cfg.get("mem_bytes", 2.4e9) / (112 * n * n))
    B = min(int(cfg.get("batch", 4096)), max(Bmax, 8))
    st32, st64 = int(cfg.get("steps32", 2500)), int(cfg.get("steps64", 600))
    topk = int(cfg.get("topk", 48))
    mu0, mu1 = float(cfg.get("mu0", 5.0)), float(cfg.get("mu1", 3e4))
    m32 = torch.triu(torch.ones(n, n, dtype=torch.float32, device=dev), 1)
    m64 = torch.triu(torch.ones(n, n, dtype=torch.float64, device=dev), 1)
    P = SumRadii(n)
    pool = mp.get_context("spawn").Pool(int(cfg.get("workers", 10)))
    best_s, best_z = -np.inf, None
    t0 = time.time()
    rnd = 0
    seen = []
    top_sols: list = []
    try:
        while time.time() - t0 < tlimit:
            rnd += 1
            x, r = init_structured(B, n, gen, dev, torch.float32, tuple(cfg.get("mix", (0.3, 0.4, 0.15, 0.15))))
            x, r = adam_optimize(x, r, m32, st32, mu0, mu1 / 10, float(cfg.get("lr0", 0.01)) / np.sqrt(n / 26), 3e-4 / np.sqrt(n / 26))
            x, r = x.double(), r.double()
            x, r = adam_optimize(x, r, m64, st64, mu1 / 10, mu1, 5e-4, 5e-5)
            sc = feasible_score(x, r, m64).cpu().numpy()
            res = polish_topk(pool, n, x.cpu().numpy(), r.cpu().numpy(), sc, topk, [])
            for s_, z_ in res:
                seen.append(s_)
                if s_ > best_s:
                    best_s, best_z = s_, z_
                if not any(abs(s_ - q[0]) < 1e-6 for q in top_sols):  # keep the best distinct basins
                    top_sols.append((s_, z_))
            top_sols = sorted(top_sols, key=lambda q: -q[0])[:int(cfg.get("keep_top", 16))]
            json.dump({"candidates": [dict(P.to_certificate(z_), score_float=float(s_)) for s_, z_ in top_sols]},
                      open(out_dir / "top.json", "w"))
            row = {"round": rnd, "t": round(time.time() - t0, 1), "B": B, "gpu_best_feas": float(sc.max()),
                   "gpu_median_feas": float(np.median(sc)), "polished_best": best_s,
                   "polished_round_top": sorted([round(s_, 6) for s_, _ in res], reverse=True)[:5],
                   "polished_round_median": float(np.median([s_ for s_, _ in res]))}
            log.write(json.dumps(row) + chr(10))
            log.flush()
            print(f"[{name}] r{rnd} t={row['t']}s B={B} gpu_best={row['gpu_best_feas']:.4f} polished_round_top="
                  f"{row['polished_round_top'][:3]} best={best_s:.6f}", flush=True)
            if best_z is not None and cfg.get("leaderboard", True):
                update_leaderboard("sum_radii", n, best_s, "gpu-large", seed, time.time() - t0, best_z, P)
    finally:
        pool.terminate()
        pool.join()
        log.close()
    summ = {"name": name, "problem": "sum_radii", "n": n, "method": "gpu-large", "seed": seed, "elapsed_s": time.time() - t0,
            "best_score": best_s, "status": classify("sum_radii", n, best_s), "record": load_record("sum_radii", n),
            "rounds": rnd, "cfg": cfg, "polished_scores_top20": sorted(seen, reverse=True)[:20]}
    json.dump(summ, open(out_dir / "summary.json", "w"), indent=1)
    if best_z is not None:
        json.dump(P.to_certificate(best_z) | {"score_float": best_s}, open(out_dir / "best.json", "w"), indent=1)
    return summ


# ------------------------------------------------------------------------------ main stage
def run(cfg: dict) -> dict:
    if cfg.get("mode") == "large":
        return run_large(cfg)
    if cfg.get("mode") == "sym":
        return run_sym(cfg)
    torch = _torch()
    from search.driver import classify, load_record, update_leaderboard
    from search.problems import SumRadii

    n = int(cfg["n"])
    mode = cfg.get("mode", "bh")
    B = int(cfg.get("batch", 4096))
    dtype = torch.float64 if cfg.get("dtype", "float32") == "float64" else torch.float32
    dev = torch.device("cuda")
    seed = int(cfg.get("seed", 0))
    tlimit = float(cfg.get("time_limit_s", 300))
    topk = int(cfg.get("topk", 48))
    name = cfg.get("name", f"gpu_{mode}_sum_radii_n{n}")
    out_dir = ROOT / "results" / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "log.jsonl", "w")
    gen = torch.Generator(device=dev).manual_seed(seed)
    mask = torch.triu(torch.ones(n, n, dtype=dtype, device=dev), 1)
    P = SumRadii(n)
    W = int(cfg.get("workers", 8))
    pool = mp.get_context("spawn").Pool(W)
    t0 = time.time()
    best_s, best_z = -np.inf, None
    polished_all: list[float] = []
    it_all = 0
    try:
        if mode == "multistart":
            steps = int(cfg.get("steps", 2500))
            rnd = 0
            while time.time() - t0 < tlimit:
                x, r = init_batch(B, n, gen, dev, dtype, cfg.get("frac_hex", 0.5))
                x, r = adam_optimize(x, r, mask, steps, cfg.get("mu0", 5.0), cfg.get("mu1", 2e4), cfg.get("lr0", 0.02),
                                     cfg.get("lr1", 2e-4))
                sc = feasible_score(x, r, mask).cpu().numpy()
                res = polish_topk(pool, n, x.cpu().numpy(), r.cpu().numpy(), sc, topk, [])
                for s, z in res:
                    polished_all.append(s)
                    if s > best_s:
                        best_s, best_z = s, z
                rnd += 1
                row = {"round": rnd, "t": round(time.time() - t0, 1), "gpu_best_feas": float(sc.max()),
                       "gpu_median_feas": float(np.median(sc)), "polished_best": best_s,
                       "polished_top": sorted([round(s, 6) for s, _ in res], reverse=True)[:5]}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"[{name}] r{rnd} t={row['t']}s gpu_best={row['gpu_best_feas']:.6f} med={row['gpu_median_feas']:.4f} "
                      f"polished_best={best_s:.9f}", flush=True)
                if best_z is not None:
                    update_leaderboard("sum_radii", n, best_s, f"gpu-{mode}", seed, time.time() - t0, best_z, P)
        else:  # parallel basin hopping
            inner = int(cfg.get("inner_steps", 350))
            init_steps = int(cfg.get("steps", 2500))
            mu0, mu1 = cfg.get("mu0", 30.0), cfg.get("mu1", 3e4)
            x, r = init_batch(B, n, gen, dev, dtype, cfg.get("frac_hex", 0.5))
            x, r = adam_optimize(x, r, mask, init_steps, 5.0, mu1, 0.02, 2e-4)
            cur = feasible_score(x, r, mask)
            T0, T1 = float(cfg.get("temp0", 0.004)), float(cfg.get("temp1", 0.0004))
            polish_every = int(cfg.get("polish_every", 40))
            it = 0
            while time.time() - t0 < tlimit:
                it += 1
                frac = (it % 300) / 300.0
                T = T0 * (T1 / T0) ** frac
                x2, r2, mv = perturb_batch(x, r, gen, mask)
                x2, r2 = adam_optimize(x2, r2, mask, inner, mu0, mu1, 0.01, 3e-4)
                new = feasible_score(x2, r2, mask)
                acc = (new >= cur) | (torch.rand(B, generator=gen, device=dev, dtype=dtype) < torch.exp((new - cur) / T))
                x = torch.where(acc[:, None, None], x2, x)
                r = torch.where(acc[:, None], r2, r)
                cur = torch.where(acc, new, cur)
                if it % polish_every == 0:
                    sc = cur.cpu().numpy()
                    res = polish_topk(pool, n, x.cpu().numpy(), r.cpu().numpy(), sc, topk, [])
                    for s, z in res:
                        polished_all.append(s)
                        if s > best_s:
                            best_s, best_z = s, z
                    row = {"iter": it, "t": round(time.time() - t0, 1), "gpu_best_feas": float(sc.max()),
                           "gpu_median_feas": float(np.median(sc)), "polished_best": best_s,
                           "polished_top": sorted([round(s, 6) for s, _ in res], reverse=True)[:5]}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(f"[{name}] it{it} t={row['t']}s gpu_best={row['gpu_best_feas']:.6f} med={row['gpu_median_feas']:.4f} "
                          f"polished_best={best_s:.9f}", flush=True)
                    if best_z is not None:
                        update_leaderboard("sum_radii", n, best_s, f"gpu-{mode}", seed, time.time() - t0, best_z, P)
                it_all = it
    finally:
        pool.terminate()
        pool.join()
        log.close()
    summ = {"name": name, "problem": "sum_radii", "n": n, "method": f"gpu-{mode}", "seed": seed,
            "elapsed_s": time.time() - t0, "best_score": best_s, "status": classify("sum_radii", n, best_s),
            "record": load_record("sum_radii", n), "iters": it_all, "cfg": cfg,
            "polished_scores_top20": sorted(polished_all, reverse=True)[:20]}
    json.dump(summ, open(out_dir / "summary.json", "w"), indent=1)
    if best_z is not None:
        json.dump(P.to_certificate(best_z) | {"score_float": best_s}, open(out_dir / "best.json", "w"), indent=1)
    return summ


# ------------------------------------------------------------------------------ symmetric (diagonal mirror) stage
def _expand_t(xf, yf, rf, td, rd):
    torch = _torch()
    x = torch.cat([xf, yf, td], -1)
    y = torch.cat([yf, xf, td], -1)
    r = torch.cat([rf, rf, rd], -1)
    return torch.stack([x, y], -1), r


def adam_sym(params, mask, steps, mu0, mu1, lr0, lr1):
    torch = _torch()
    ps = [t.clone().requires_grad_(True) for t in params]
    opt = torch.optim.Adam(ps, lr=lr0)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        mu = mu0 * (mu1 / mu0) ** f
        for g in opt.param_groups:
            g["lr"] = lr0 * (lr1 / lr0) ** f
        opt.zero_grad(set_to_none=True)
        x, r = _expand_t(*ps)
        loss = (-r.sum(1) + mu * penalty(x, r, mask)).sum()
        loss.backward()
        opt.step()
    return [t.detach() for t in ps]


def _polish_sym_one(args):
    from search.large import slp_local
    from search.problems import SumRadii
    from search.symm import expand, slp_local_sym

    n, p, q, zr = args
    zr2, _ = slp_local_sym(zr, p, q)
    z_sym = expand(zr2, p, q)
    P = SumRadii(n)
    res_sym = float(z_sym[2 * n :].sum())
    z2, _ = slp_local(z_sym, n)  # relax the symmetry constraint
    r = P.lp_radii(z2[:n], z2[n : 2 * n])
    zf = np.concatenate([z2[: 2 * n], r])
    return float(r.sum()), zf, res_sym


def run_sym(cfg: dict) -> dict:
    torch = _torch()
    from search.driver import classify, load_record, update_leaderboard
    from search.problems import SumRadii

    n = int(cfg["n"])
    name = cfg.get("name", f"gpu_sym_sum_radii_n{n}")
    out_dir = ROOT / "results" / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "log.jsonl", "a")
    seed = int(cfg.get("seed", 0))
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)
    tlimit = float(cfg.get("time_limit_s", 600))
    Bmax = int(cfg.get("mem_bytes", 2.4e9) / (112 * n * n))
    B = min(int(cfg.get("batch", 2048)), max(Bmax, 8))
    st32, st64 = int(cfg.get("steps32", 2500)), int(cfg.get("steps64", 600))
    topk = int(cfg.get("topk", 48))
    mu0, mu1 = float(cfg.get("mu0", 5.0)), float(cfg.get("mu1", 3e4))
    m32 = torch.triu(torch.ones(n, n, dtype=torch.float32, device=dev), 1)
    m64 = torch.triu(torch.ones(n, n, dtype=torch.float64, device=dev), 1)
    P = SumRadii(n)
    pool = mp.get_context("spawn").Pool(int(cfg.get("workers", 8)))
    qs = cfg.get("q_values") or [q for q in range(n % 2, int(2 * np.sqrt(n)) + 2, 2)]
    best_s, best_z = -np.inf, None
    t0 = time.time()
    rnd = 0
    seen: list[float] = []
    top_sols: list = []
    try:
        while time.time() - t0 < tlimit:
            rnd += 1
            q = int(qs[(rnd - 1) % len(qs)])
            p = (n - q) // 2
            r0 = 0.4 / np.sqrt(n)
            xf = torch.rand(B, p, generator=gen, device=dev, dtype=torch.float32)
            yf = torch.rand(B, p, generator=gen, device=dev, dtype=torch.float32)
            td = torch.rand(B, q, generator=gen, device=dev, dtype=torch.float32)
            rf = torch.full((B, p), r0, device=dev, dtype=torch.float32)
            rd = torch.full((B, q), r0, device=dev, dtype=torch.float32)
            lr0 = float(cfg.get("lr0", 0.01)) / np.sqrt(n / 26)
            ps = adam_sym([xf, yf, rf, td, rd], m32, st32, mu0, mu1 / 10, lr0, 3e-4 / np.sqrt(n / 26))
            ps = [t.double() for t in ps]
            ps = adam_sym(ps, m64, st64, mu1 / 10, mu1, 5e-4, 5e-5)
            x, r = _expand_t(*ps)
            sc = feasible_score(x, r, m64).cpu().numpy()
            order = np.argsort(-sc)
            picked, last = [], []
            xs, ys, rs = [t.cpu().numpy() for t in ps[:3]], None, None
            for i in order:
                if any(abs(sc[i] - t) < 2e-6 for t in last):
                    continue
                last.append(float(sc[i]))
                zr = np.concatenate([ps[0][i].cpu().numpy(), ps[1][i].cpu().numpy(), np.maximum(ps[2][i].cpu().numpy(), 1e-4),
                                     ps[3][i].cpu().numpy(), np.maximum(ps[4][i].cpu().numpy(), 1e-4)])
                picked.append((n, p, q, zr))
                if len(picked) >= topk:
                    break
            res = pool.map(_polish_sym_one, picked, chunksize=1)
            for s_, z_, _ in res:
                seen.append(s_)
                if s_ > best_s:
                    best_s, best_z = s_, z_
                if not any(abs(s_ - t[0]) < 1e-6 for t in top_sols):
                    top_sols.append((s_, z_))
            top_sols = sorted(top_sols, key=lambda t: -t[0])[: int(cfg.get("keep_top", 16))]
            json.dump({"candidates": [dict(P.to_certificate(z_), score_float=float(s_)) for s_, z_ in top_sols]},
                      open(out_dir / "top.json", "w"))
            row = {"round": rnd, "t": round(time.time() - t0, 1), "q": q, "p": p, "B": B, "gpu_best_feas": float(sc.max()),
                   "polished_best": best_s, "polished_round_top": sorted([round(s_, 6) for s_, _, _ in res], reverse=True)[:5],
                   "sym_only_top": sorted([round(t, 6) for _, _, t in res], reverse=True)[:3]}
            log.write(json.dumps(row) + chr(10))
            log.flush()
            print(f"[{name}] r{rnd} q={q} t={row['t']}s gpu_best={row['gpu_best_feas']:.4f} top={row['polished_round_top'][:3]} best={best_s:.6f}", flush=True)
            if best_z is not None and cfg.get("leaderboard", True):
                update_leaderboard("sum_radii", n, best_s, "gpu-sym", seed, time.time() - t0, best_z, P)
    finally:
        pool.terminate()
        pool.join()
        log.close()
    summ = {"name": name, "problem": "sum_radii", "n": n, "method": "gpu-sym", "seed": seed, "elapsed_s": time.time() - t0,
            "best_score": best_s, "status": classify("sum_radii", n, best_s), "record": load_record("sum_radii", n),
            "rounds": rnd, "cfg": cfg, "polished_scores_top20": sorted(seen, reverse=True)[:20]}
    json.dump(summ, open(out_dir / "summary.json", "w"), indent=1)
    if best_z is not None:
        json.dump(P.to_certificate(best_z) | {"score_float": best_s}, open(out_dir / "best.json", "w"), indent=1)
    return summ


if __name__ == "__main__":
    mp.freeze_support()
    cfg = yaml.safe_load(open(sys.argv[1]))
    for c in (cfg if isinstance(cfg, list) else [cfg]):
        print(json.dumps({k: v for k, v in run(c).items() if k != "cfg"}, indent=1))
