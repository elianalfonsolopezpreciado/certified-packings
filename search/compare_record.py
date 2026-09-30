"""Symmetry-aware structural comparison of our sum-of-radii certificate with Packomania's listed csqv configuration.

    python -m search.compare_record 250          # uses data/packomania_files/csqv250.txt (download it with scripts/fetch_packomania.py; not redistributed) and certificates/A_n250.json

For each of the 8 symmetries of the square we solve the optimal one-to-one assignment (Hungarian algorithm) between our circles and the
listed circles, with cost = Euclidean distance in (x, y, r), and report the deviations of the best symmetry. Interpretation used in the report:
  * max deviation <= 1e-6 (and equal contact structure): the SAME packing (differences are numerical, e.g. a safety margin such as n=27),
  * large deviations for many circles: a DIFFERENT structure.
It also recomputes the exact sum of the listed radii (must equal the listed 'sumradii' up to 12-decimal rounding) and reports the gap.
Writes results/structure_compare_n<n>.txt.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]


def load_listed(n: int):
    if not (ROOT / "data" / "packomania_files" / f"csqv{n}.txt").exists():
        raise SystemExit(f"missing data/packomania_files/csqv{n}.txt: run `python scripts/fetch_packomania.py --n {n}` first (Packomania files are not redistributed)")
    rows = []
    for line in open(ROOT / "data" / "packomania_files" / f"csqv{n}.txt"):
        if line.startswith("#") or not line.strip():
            continue
        a = line.split()
        rows.append((F(a[1]) + F(1, 2), F(a[2]) + F(1, 2), F(a[3])))
    listed_sum_line = [l for l in open(ROOT / "data" / "packomania_files" / f"csqv{n}.txt") if "sumradii" in l][0].split("=")[1].strip()
    return rows, listed_sum_line


def main(n: int) -> str:
    rows, sum_line = load_listed(n)
    assert len(rows) == n
    P = np.array([[float(a), float(b), float(c)] for a, b, c in rows])
    ours = json.load(open(ROOT / "certificates" / (f"A_n{n}.json" if n >= 100 else f"sum_radii_n{n}.json")))
    O = np.array([[float(v) for v in c] for c in ours["circles"]])
    listed_sum = sum(r for _, _, r in rows)
    out_lines = []
    best = None
    for swap in (0, 1):
        for fx in (0, 1):
            for fy in (0, 1):
                Q = P.copy()
                if swap:
                    Q[:, [0, 1]] = Q[:, [1, 0]]
                if fx:
                    Q[:, 0] = 1 - Q[:, 0]
                if fy:
                    Q[:, 1] = 1 - Q[:, 1]
                C = np.linalg.norm(O[:, None, :] - Q[None, :, :], axis=2)
                ri, ci = linear_sum_assignment(C)
                dev = np.abs(O[ri] - Q[ci]).max(1)
                stat = {"sym": (swap, fx, fy), "max": float(dev.max()), "median": float(np.median(dev)), "within_1e-6": int((dev < 1e-6).sum()),
                        "within_1e-3": int((dev < 1e-3).sum()), "mean_dr": float(np.mean(O[ri, 2] - Q[ci, 2]))}
                out_lines.append(f"symmetry (swap,flipx,flipy)={stat['sym']}: max dev {stat['max']:.3e}, median {stat['median']:.3e}, "
                                 f"within 1e-6: {stat['within_1e-6']}/{n}, within 1e-3: {stat['within_1e-3']}/{n}")
                if best is None or (stat["within_1e-3"], -stat["median"]) > (best["within_1e-3"], -best["median"]):
                    best = stat
                    best_match = (Q.copy(), ri.copy(), ci.copy(), dev.copy())
    ours_raw = float(O[:, 2].sum())
    verdict = "SAME packing (numerical/safety-margin differences only)" if best["max"] < 1e-6 else "DIFFERENT structure"
    header = [f"n={n}: listed sumradii (file) = {sum_line}; exact sum of the listed radii = {float(listed_sum):.12f}; our raw exact sum (float view) = {ours_raw:.12f}",
              f"gap (ours raw - listed sum of radii) = {ours_raw - float(listed_sum):+.6e}",
              f"BEST symmetry {best['sym']}: max coordinate/radius deviation {best['max']:.3e}, median {best['median']:.3e}, circles within 1e-6: {best['within_1e-6']}/{n}, "
              f"within 1e-3: {best['within_1e-3']}/{n}, mean(our r - listed r) = {best['mean_dr']:+.3e}",
              f"VERDICT: {verdict}", "", "all 8 symmetries:"]
    text = "\n".join(header + out_lines) + "\n"
    (ROOT / "results" / f"structure_compare_n{n}.txt").write_text(text, encoding="utf-8")
    print(text)
    _overlay(n, O, best_match)
    return verdict


def _overlay(n, O, best_match) -> None:
    """results/plots/overlay_n<n>.png: listed circles (outline) vs ours (filled, colour = deviation), best symmetry."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    Q, ri, ci, dev = best_match
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=1.0))
    for j in range(len(Q)):
        ax.add_patch(Circle((Q[j, 0], Q[j, 1]), Q[j, 2], fill=False, ec="k", lw=0.5))
    cmap = plt.get_cmap("viridis")
    lo = np.log10(np.maximum(dev, 1e-9))
    for k, (i, j) in enumerate(zip(ri, ci)):
        c = cmap((np.clip(lo[k], -6, -1) + 6) / 5)
        ax.add_patch(Circle((O[i, 0], O[i, 1]), O[i, 2], fill=True, fc=c, alpha=0.45, ec="none"))
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.set_title(f"n={n}: ours (filled, colour = log10 deviation from best-matched listed circle) vs listed (outline)", fontsize=7)
    fig.tight_layout()
    (ROOT / "results" / "plots").mkdir(exist_ok=True)
    fig.savefig(ROOT / "results" / "plots" / f"overlay_n{n}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main(int(sys.argv[1]))
