"""Tests for the large-n machinery: neighbour lists, sparse SLP, mixed-precision polish, verifier on n ~ 100+."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from search.large import TR_FACTOR, all_pairs_max_violation, neighbor_pairs, slp_local, slp_step
from search.polish import polish, polish_mixed
from search.problems import SumRadii
from verify import parse, verify_fraction, verify_mpmath

ROOT = Path(__file__).resolve().parents[1]


def _n26_z():
    c = json.load(open(ROOT / "certificates" / "sum_radii_n26.json"))
    a = np.array([[float(v) for v in row] for row in c["circles"]])
    return np.concatenate([a[:, 0], a[:, 1], a[:, 2]])


def test_neighbor_pairs_equal_bruteforce():
    rng = np.random.default_rng(0)
    n = 80
    P = SumRadii(n)
    z = P.local_opt(P.hex_init(rng))  # n >= 60 -> SLP path
    x, y, r = z.z[:n], z.z[n : 2 * n], z.z[2 * n :]
    for margin in (0.0, 1e-3, 0.05):
        I, J = neighbor_pairs(x, y, r, margin)
        got = set(zip(I.tolist(), J.tolist()))
        iu, ju = np.triu_indices(n, 1)
        gap = np.hypot(x[iu] - x[ju], y[iu] - y[ju]) - (r[iu] + r[ju])
        want = set(zip(iu[gap < margin].tolist(), ju[gap < margin].tolist()))
        assert got == want


def test_slp_step_never_violates_pairs_outside_neighbor_list():
    """The neighbour cutoff TR_FACTOR*Delta is exact: after a step, brute-force violation over ALL pairs is ~0."""
    rng = np.random.default_rng(1)
    n = 90
    P = SumRadii(n)
    z0 = P.hex_init(rng)
    x, y, r = z0[:n].copy(), z0[n : 2 * n].copy(), z0[2 * n :].copy() * 0.6  # loose start: big steps possible
    for delta in (0.002, 0.02, 0.05):
        st = slp_step(x, y, r, delta)
        assert st is not None
        dx, dy, dr = st
        assert max(np.abs(dx).max(), np.abs(dy).max(), np.abs(dr).max()) <= delta + 1e-9
        assert all_pairs_max_violation(x + dx, y + dy, r + dr) < 1e-8
    assert TR_FACTOR > 2 * np.sqrt(2) + 1.99


def test_slp_is_monotone_feasible_and_returns_to_record_basin():
    z = _n26_z()
    n = 26
    rng = np.random.default_rng(2)
    zp = z + rng.normal(0, 1e-3, len(z))
    zp[2 * n :] = np.maximum(zp[2 * n :] - 5e-3, 1e-3)  # start strictly feasible-ish and clearly below the optimum
    out, its = slp_local(zp, n)
    x, y, r = out[:n], out[n : 2 * n], out[2 * n :]
    assert all_pairs_max_violation(x, y, r) < 1e-9
    assert r.sum() == pytest.approx(2.635983084917608, abs=5e-9)


def test_local_opt_dispatches_to_slp_for_large_n_and_is_feasible():
    n = 100
    P = SumRadii(n)
    res = P.local_opt(P.hex_init(np.random.default_rng(3)))
    assert res.raw_violation < 1e-9
    assert 5.0 < res.score < 5.3  # Packomania record at n=100 is 5.2637


def test_polish_mixed_matches_dense_polish():
    z = _n26_z() + np.random.default_rng(4).normal(0, 1e-11, 78)
    a = polish("sum_radii", 26, z)
    b = polish_mixed(26, z)
    assert abs(float(a["info"]["score_mp"]) - float(b["info"]["score_mp"])) < 1e-25
    assert b["info"]["max_active_residual"] < 1e-45


def _big_certificate(n=120, seed=0):
    P = SumRadii(n)
    res = P.local_opt(P.hex_init(np.random.default_rng(seed)))
    pol = polish_mixed(n, res.z)
    return json.loads(json.dumps(pol["certificate"]), parse_float=str)


def test_verifier_certifies_polished_n120_and_backends_agree():
    cert = _big_certificate()
    parsed = parse(cert)
    a, b = verify_fraction(parsed), verify_mpmath(parsed)
    assert a.valid and b.valid
    assert a.certified_score[:20] == b.certified_score[:20]


def test_verifier_checks_all_pairs_not_just_neighbours():
    """An overlap between circles with far-apart INDICES and far-apart in space is still caught (no neighbour lists)."""
    cert = _big_certificate()
    cert["circles"][117] = list(cert["circles"][3])  # duplicate circle 3 at index 117 -> overlap
    parsed = parse(cert)
    assert not verify_fraction(parsed).valid
    assert not verify_mpmath(parsed).valid


def test_convert_to_n_insert_and_delete():
    from search.methods import convert_to_n

    rng = np.random.default_rng(7)
    P100, P101, P99 = SumRadii(100), SumRadii(101), SumRadii(99)
    z100 = P100.local_opt(P100.hex_init(rng)).z
    up = convert_to_n(P101, z100, 100, rng)  # insert a circle in the largest void
    assert up.shape == (3 * 101,) and P101.max_violation(up) < 1e-7 and up[2 * 101 :].min() >= 0
    down = convert_to_n(P99, z100, 100, rng)  # delete the smallest circle
    assert down.shape == (3 * 99,) and P99.max_violation(down) < 1e-7
    assert down[2 * 99 :].sum() >= z100[2 * 100 :].sum() - z100[2 * 100 :].min() - 1e-9
    assert convert_to_n(P99, z100, 90, rng) is None  # only n, n+-1 are converted
