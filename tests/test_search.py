"""Tests for solvers, moves, the high-precision polish and seedable/resumable runs."""
from __future__ import annotations

import json
import shutil
from fractions import Fraction

import numpy as np
import pytest

from search.methods import crossover, perturb
from search.polish import polish
from search.problems import MinDistance, SumRadii
from verify import parse, verify_fraction


def test_sum_radii_local_opt_feasible_and_lp_exact():
    P = SumRadii(6)
    rng = np.random.default_rng(0)
    res = P.local_opt(P.random_init(rng))
    assert P.max_violation(res.z) < 1e-9
    # LP radii for fixed centres are optimal: no other feasible radii vector is better
    x, y, r = P.split(res.z)
    r2 = P.lp_radii(x, y)
    assert abs(r2.sum() - res.score) < 1e-9


N4_RECORD = 1.006788474668  # Packomania csqv, n=4 (Packomania csqv sumradii.txt, 2026-09-29); NOT 1.0 (equal circles)


def test_four_circles_reaches_packomania_record():
    P = SumRadii(4)
    z0 = P.from_centers(np.array([0.3, 0.7, 0.3, 0.7, 0.3, 0.3, 0.7, 0.7]))
    res = P.local_opt(z0)
    assert res.score == pytest.approx(N4_RECORD, abs=5e-12)


def test_min_distance_n5_is_sqrt2_over_2():
    P = MinDistance(5)
    best = 0.0
    rng = np.random.default_rng(3)
    for _ in range(20):
        best = max(best, P.local_opt(P.random_init(rng)).score)
    assert best == pytest.approx(np.sqrt(2) / 2, abs=1e-9)


def test_polish_gives_closed_form_min_distance_n5():
    P = MinDistance(5)
    rng = np.random.default_rng(3)
    best = max((P.local_opt(P.random_init(rng)) for _ in range(20)), key=lambda r: r.score)
    out = polish("min_distance", 5, best.z)
    assert out["info"]["max_active_residual"] < 1e-40
    rep = verify_fraction(parse(json.loads(json.dumps(out["certificate"]), parse_float=str)))
    assert rep.valid
    exact = np.sqrt(2) / 2
    assert abs(float(rep.certified_score) - exact) < 1e-14


def test_polish_sum_radii_n4_is_strictly_feasible_and_matches_record():
    P = SumRadii(4)
    res = P.local_opt(P.from_centers(np.array([0.3, 0.7, 0.3, 0.7, 0.3, 0.3, 0.7, 0.7])))
    out = polish("sum_radii", 4, res.z)
    cert = json.loads(json.dumps(out["certificate"]), parse_float=str)
    rep = verify_fraction(parse(cert))
    assert rep.valid
    # certified score = raw optimum - 4*eps, raw optimum equals the 12-decimal record
    raw = Fraction(rep.certified_score) + 4 * Fraction(1, 10**12)
    assert abs(float(raw) - N4_RECORD) < 5e-12


def test_moves_keep_shapes_and_bounds():
    rng = np.random.default_rng(1)
    for P in (SumRadii(9), MinDistance(9)):
        z = P.local_opt(P.random_init(rng)).z
        for mv in ("jitter_small", "jitter_big", "reinsert_k", "relocate_one"):
            z2, _ = perturb(P, z, rng, mv)
            assert z2.shape == z.shape
            assert np.all(z2[: 2 * P.n] >= 0) and np.all(z2[: 2 * P.n] <= 1)
        z3 = crossover(P, z, P.random_init(rng), rng)
        assert z3.shape == z.shape


def test_resume_is_deterministic(tmp_path, monkeypatch):
    from search import driver

    monkeypatch.setattr(driver, "RESULTS", tmp_path)
    base = dict(problem="min_distance", n=8, method="basin_hopping", seed=5, workers=2, iters_per_round=4,
                time_limit_s=1000, leaderboard=False)
    a = driver.run_experiment(dict(base, name="a", max_rounds=2), quiet=True)
    driver.run_experiment(dict(base, name="b", max_rounds=1), quiet=True)
    b = driver.run_experiment(dict(base, name="b", max_rounds=2), quiet=True)  # resumes from round 1
    assert b["rounds"] == 2
    assert a["best_score"] == pytest.approx(b["best_score"], abs=1e-12)
    shutil.rmtree(tmp_path, ignore_errors=True)
