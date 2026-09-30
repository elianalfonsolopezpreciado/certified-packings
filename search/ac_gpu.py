"""GPU minimax optimiser for the step-function autoconvolution problem (problem C), used by search/ac_c2f.py.

Objective (scale invariant): normalise a = N * softmax(theta) (mean height 1, a > 0), c = a*a via float64 FFT,
    R(a) = 2 * max_m c_m / N            (see verify/core.py for the exact definition of the certified ratio)
Smooth surrogate: R_tau = tau * log sum_m exp(2 c_m / (N tau)), tau annealed from tau0 to tau1.
Optimisers: Adam (few hundred steps per tau) or L-BFGS (strong Wolfe). Floats are only used for search: the final
ratio is always re-computed exactly by the verifier.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402


def _torch():
    import torch

    return torch


def heights(theta):
    torch = _torch()
    n = theta.shape[-1]
    return n * torch.softmax(theta, dim=-1)


def autoconv_ratio_terms(a):
    """c = a*a (length 2N-1) via float64 FFT; a is [B, N]."""
    torch = _torch()
    n = a.shape[-1]
    fa = torch.fft.rfft(a, n=2 * n)
    return torch.fft.irfft(fa * fa, n=2 * n)[..., : 2 * n - 1]


def true_ratio(a):
    n = a.shape[-1]
    return 2.0 * autoconv_ratio_terms(a).amax(-1) / n


def soft_loss(theta, tau):
    torch = _torch()
    n = theta.shape[-1]
    a = heights(theta)
    c = 2.0 * autoconv_ratio_terms(a) / n
    m = c.amax(-1, keepdim=True).detach()
    return (m.squeeze(-1) + tau * torch.log(torch.exp((c - m) / tau).sum(-1))), c


def optimize(theta0: np.ndarray, taus: list[float], adam_steps: int = 300, lbfgs_iters: int = 0, lr: float = 0.02,
             device: str = "cuda"):
    """theta0: [B, N] float64 numpy (log-heights). Returns (theta, ratio) numpy arrays.

    For each tau (descending): Adam for `adam_steps`, then optionally `lbfgs_iters` L-BFGS iterations (batch of 1 only)."""
    torch = _torch()
    dev = torch.device(device)
    th = torch.tensor(theta0, dtype=torch.float64, device=dev)
    best_th = th.clone()
    best_r = true_ratio(heights(th)).detach()
    for tau in taus:
        th = best_th.clone().requires_grad_(True)  # restart each stage from the best-so-far (monotone)
        opt = torch.optim.Adam([th], lr=lr)
        for s in range(adam_steps):
            for g in opt.param_groups:
                g["lr"] = lr * (0.1 ** (s / max(adam_steps - 1, 1)))
            opt.zero_grad(set_to_none=True)
            loss, _ = soft_loss(th, tau)
            loss.sum().backward()
            opt.step()
        if lbfgs_iters and th.shape[0] == 1:
            lb = torch.optim.LBFGS([th], lr=1.0, max_iter=lbfgs_iters, history_size=20, line_search_fn="strong_wolfe",
                                   tolerance_grad=1e-14, tolerance_change=1e-16)

            def closure():
                lb.zero_grad()
                l, _ = soft_loss(th, tau)
                l.sum().backward()
                return l.sum()

            lb.step(closure)
        with torch.no_grad():
            r = true_ratio(heights(th))
            better = r < best_r
            best_th = torch.where(better[:, None], th.detach(), best_th)
            best_r = torch.where(better, r, best_r)
    return best_th.cpu().numpy(), best_r.cpu().numpy()


def theta_from_heights(a: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    a = np.maximum(np.asarray(a, dtype=np.float64), floor * float(np.mean(a)))
    return np.log(a)


def heights_from_theta(theta: np.ndarray) -> np.ndarray:
    t = theta - theta.max(-1, keepdims=True)
    e = np.exp(t)
    return theta.shape[-1] * e / e.sum(-1, keepdims=True)


def homotopy(theta0: np.ndarray, tau0: float, tau1: float, stages: int, steps: int, lr: float = 0.03,
             lr_end: float = 0.003, device: str = "cuda", sym: bool = False):
    """Track the minimiser of the smooth surrogate while tau decays geometrically tau0 -> tau1.

    theta0 [B, N]. Adam with continuous state, `steps` steps per stage. Keeps the best true ratio seen per row."""
    torch = _torch()
    dev = torch.device(device)
    t0_ = torch.tensor(theta0, dtype=torch.float64, device=dev)
    if sym:  # optimise only the left half; the profile is mirrored (a_i = a_{N-1-i}); theta0 is symmetrised first
        half = t0_.shape[-1] // 2
        t0_ = 0.5 * (t0_[..., :half] + t0_.flip(-1)[..., :half])
    expand = (lambda h: torch.cat([h, h.flip(-1)], dim=-1)) if sym else (lambda h: h)
    th = t0_.requires_grad_(True)
    opt = torch.optim.Adam([th], lr=lr)
    best_th = th.detach().clone()
    best_r = true_ratio(heights(expand(th))).detach()
    total = stages * steps
    k = 0
    for st in range(stages):
        tau = tau0 * (tau1 / tau0) ** (st / max(stages - 1, 1))
        for _ in range(steps):
            f = k / max(total - 1, 1)
            for g in opt.param_groups:
                g["lr"] = lr * (lr_end / lr) ** f
            opt.zero_grad(set_to_none=True)
            loss, _ = soft_loss(expand(th), tau)
            loss.sum().backward()
            opt.step()
            k += 1
        with torch.no_grad():
            r = true_ratio(heights(expand(th)))
            better = r < best_r
            best_th = torch.where(better[:, None], th.detach(), best_th)
            best_r = torch.where(better, r, best_r)
    return expand(best_th).cpu().numpy(), best_r.cpu().numpy()
