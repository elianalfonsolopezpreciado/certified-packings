"""Unit tests for the independent verifier, including adversarial certificates."""
from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from verify import CertificateError, load_certificate, parse, verify_fraction, verify_mpmath

ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path, obj) -> str:
    p = tmp_path / "c.json"
    p.write_text(json.dumps(obj))
    return str(p)


def _both(obj, eps=Fraction(1, 10**12)):
    parsed = parse(json.loads(json.dumps(obj), parse_float=str))
    return verify_fraction(parsed, eps), verify_mpmath(parsed, eps)


def sr(circles):
    return {"problem": "sum_radii", "n": len(circles), "circles": circles}


def md(points):
    return {"problem": "min_distance", "n": len(points), "points": points}


# ------------------------------------------------------------------ valid cases
def test_single_circle_fills_square():
    a, b = _both(sr([["0.5", "0.5", "0.5"]]))
    assert a.valid and b.valid
    assert a.certified_score.startswith("0.49999999999")  # 0.5 - 1e-12
    assert Fraction(a.raw_score) == Fraction(1, 2)


def test_tangent_circles_2x2_grid_valid_even_without_shrink():
    circles = [[x, y, "0.25"] for x in ("0.25", "0.75") for y in ("0.25", "0.75")]
    for eps in (Fraction(0), Fraction(1, 10**12)):
        a, b = _both(sr(circles), eps)
        assert a.valid and b.valid
        assert Fraction(a.raw_max_violation) == 0  # exactly tangent, exactly zero violation


def test_score_is_sum_of_shrunken_radii_with_many_digits():
    circles = [[x, y, "0.25"] for x in ("0.25", "0.75") for y in ("0.25", "0.75")]
    a, _ = _both(sr(circles))
    assert abs(Fraction(a.certified_score) - (1 - 4 * Fraction(1, 10**12))) < Fraction(1, 10**20)
    assert len(a.certified_score.replace(".", "")) >= 20


def test_decimal_text_parsed_exactly_not_via_float():
    # 0.1 has no exact binary float; the verifier must treat the text "0.1" as exactly 1/10.
    circles = [["0.1", "0.1", "0.1"], ["0.3", "0.1", "0.1"]]
    a, b = _both(sr(circles), Fraction(0))
    assert a.valid and b.valid and Fraction(a.raw_max_violation) == 0


# ------------------------------------------------------------------ adversarial cases
def test_circle_outside_boundary_rejected():
    a, b = _both(sr([["0.5", "0.5", "0.6"]]))
    assert not a.valid and not b.valid
    a, b = _both(sr([["1.2", "0.5", "0.01"]]))
    assert not a.valid and not b.valid


def test_circle_poking_out_by_tiny_amount_rejected():
    # pokes out by 1e-9 > shrink 1e-12
    a, b = _both(sr([["0.5", "0.5", "0.500000001"]]))
    assert not a.valid and not b.valid


def test_overlap_beyond_epsilon_rejected():
    circles = [["0.25", "0.25", "0.25000001"], ["0.75", "0.25", "0.25"]]
    a, b = _both(sr(circles))
    assert not a.valid and not b.valid


def test_overlap_within_shrink_is_repaired_and_flagged_raw_infeasible():
    # overlap of 1e-13 in radius terms: infeasible raw, feasible after the documented shrink.
    circles = [["0.25", "0.25", "0.25"], ["0.75", "0.25", "0.2500000000001"]]
    a, b = _both(sr(circles), Fraction(1, 10**12))
    assert a.valid and b.valid
    assert Fraction(a.raw_max_violation) > 0
    a0, b0 = _both(sr(circles), Fraction(0))
    assert not a0.valid and not b0.valid


def test_duplicate_circles_rejected():
    a, b = _both(sr([["0.5", "0.5", "0.1"], ["0.5", "0.5", "0.1"]]))
    assert not a.valid and not b.valid


def test_negative_radius_rejected():
    a, b = _both(sr([["0.5", "0.5", "-0.1"]]))
    assert not a.valid and not b.valid


def test_nonfinite_and_malformed_inputs_raise():
    for bad in ("nan", "inf", "-inf", "abc", ""):
        with pytest.raises(CertificateError):
            parse({"problem": "sum_radii", "circles": [["0.5", "0.5", bad]]})
    with pytest.raises(CertificateError):
        parse({"problem": "sum_radii", "n": 3, "circles": [["0.5", "0.5", "0.1"]]})
    with pytest.raises(CertificateError):
        parse({"problem": "nope", "circles": []})
    with pytest.raises(CertificateError):
        parse({"problem": "sum_radii", "circles": [["0.5", "0.5"]]})
    with pytest.raises(CertificateError):
        parse({"problem": "sum_radii", "circles": [[True, "0.5", "0.1"]]})


def test_nan_json_constant_rejected(tmp_path):
    p = tmp_path / "n.json"
    p.write_text('{"problem":"sum_radii","circles":[[0.5,0.5,NaN]]}')
    with pytest.raises(CertificateError):
        load_certificate(str(p))


# ------------------------------------------------------------------ min_distance
def test_min_distance_unit_square_corners():
    pts = [["0", "0"], ["1", "0"], ["0", "1"], ["1", "1"]]
    a, b = _both(md(pts))
    assert a.valid and b.valid
    assert Fraction(a.certified_score) == 1 or abs(float(a.certified_score) - 1.0) < 1e-25


def test_min_distance_duplicate_points_gives_zero():
    a, b = _both(md([["0.2", "0.2"], ["0.2", "0.2"], ["0.9", "0.9"]]))
    assert a.valid and float(a.certified_score) == 0.0
    assert float(b.certified_score) == 0.0


def test_min_distance_point_outside_rejected():
    for p in ([["0", "0"], ["1.0000000000001", "0.5"]], [["-1e-30", "0"], ["1", "1"]]):
        a, b = _both(md(p))
        assert not a.valid and not b.valid


def test_min_distance_sqrt_precision():
    a, b = _both(md([["0", "0"], ["1", "1"]]))
    assert a.certified_score.startswith("1.41421356237309504880168872")
    assert b.certified_score.startswith("1.41421356237309504880168872")


def test_backends_agree_on_random_configs():
    import random

    rng = random.Random(1)
    for _ in range(5):
        pts = [[f"{rng.random():.15f}", f"{rng.random():.15f}"] for _ in range(12)]
        a, b = _both(md(pts))
        assert a.valid and b.valid
        assert a.certified_score[:20] == b.certified_score[:20]


# ------------------------------------------------------------------ CLI
def test_cli_exit_codes(tmp_path):
    good = _write(tmp_path, sr([["0.5", "0.5", "0.5"]]))
    r = subprocess.run([sys.executable, "-m", "verify", good], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    bad = _write(tmp_path, sr([["0.5", "0.5", "0.7"]]))
    r = subprocess.run([sys.executable, "-m", "verify", bad], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1
    p = tmp_path / "garbage.json"
    p.write_text("{not json")
    r = subprocess.run([sys.executable, "-m", "verify", str(p)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2


def test_verifier_does_not_import_search_code():
    import re

    for f in (ROOT / "verify").glob("*.py"):
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+search", text, re.M), f


# ------------------------------------------------------------------ autocorr1 (problem C)
def ac(heights):
    return {"problem": "autocorr1", "n": len(heights), "heights": heights}


def test_autocorr1_uniform_is_two_and_half_indicator_is_four():
    a, b = _both(ac(["1"] * 7))
    assert a.valid and b.valid and Fraction(a.certified_score) == 2
    a, b = _both(ac(["1", "0"]))  # indicator of half the support: peak (f*f)=1/4, (int f)^2=1/16
    assert Fraction(a.certified_score) == 4
    assert float(b.certified_score) == pytest.approx(4.0, abs=1e-25)


def test_autocorr1_matches_numpy_on_random_step_functions():
    import numpy as np

    rng = np.random.default_rng(0)
    for n in (5, 33):
        a = rng.random(n)
        rep_a, rep_b = _both(ac([f"{v:.15f}" for v in a]))
        aa = np.array([float(f"{v:.15f}") for v in a])
        ref = 2 * n * np.convolve(aa, aa).max() / aa.sum() ** 2
        assert float(rep_a.certified_score) == pytest.approx(ref, rel=1e-13)
        assert rep_a.certified_score[:18] == rep_b.certified_score[:18]


def test_autocorr1_rejects_negative_zero_and_nonfinite():
    a, b = _both(ac(["1", "-0.001", "1"]))
    assert not a.valid and not b.valid
    a, b = _both(ac(["0", "0"]))
    assert not a.valid and not b.valid
    with pytest.raises(CertificateError):
        parse({"problem": "autocorr1", "heights": ["1", "nan"]})


# ------------------------------------------------------------------ large-N exact autoconvolution (verify/ac1.py)
def _rand_ints(n, bits, seed, zeros=0.2):
    import random

    rng = random.Random(seed)
    return [0 if rng.random() < zeros else rng.getrandbits(bits) for _ in range(n)]


def test_ac1_three_algorithms_agree_exactly():
    from verify import ac1

    for n, bits in ((1, 10), (2, 64), (37, 57), (150, 90), (400, 57)):
        A = _rand_ints(n, bits, n)
        if not any(A):
            A[0] = 1
        d = ac1.conv_direct(A)
        assert ac1.conv_kronecker(A) == d
        assert ac1.conv_limbs(A) == d


def test_ac1_adversarial_all_max_and_single_spike():
    from verify import ac1

    A = [2**60 - 1] * 300  # maximal slot fill: catches carries between Kronecker slots
    assert ac1.conv_kronecker(A) == ac1.conv_direct(A) == ac1.conv_limbs(A)
    B = [0] * 299 + [7]
    assert ac1.max_kronecker(B) == ac1.max_limbs(B) == 49


def test_autocorr1_large_n_backends_agree_and_match_numpy_to_float_precision():
    import numpy as np

    rng = np.random.default_rng(5)
    n = 700  # > SMALL_N: fraction backend uses Kronecker, mpmath backend uses int64 limbs
    a = rng.random(n) ** 3
    a[rng.random(n) < 0.3] = 0.0
    a[0] = 1.0
    rep_a, rep_b = _both(ac([f"{v:.15f}" for v in a]))
    assert rep_a.valid and rep_b.valid
    assert rep_a.certified_score == rep_b.certified_score
    aa = np.array([float(f"{v:.15f}") for v in a])
    assert float(rep_a.certified_score) == pytest.approx(2 * n * np.convolve(aa, aa).max() / aa.sum() ** 2, rel=1e-12)


def test_autocorr1_gz_certificate_roundtrip(tmp_path):
    import gzip

    obj = ac(["1", "0.5", "0", "2.25"])
    p = tmp_path / "c.json.gz"
    with gzip.open(p, "wt") as fh:
        json.dump(obj, fh)
    from verify import verify_file

    a = verify_file(str(p), "fraction")
    b = verify_file(str(p), "mpmath")
    assert a.valid and a.certified_score == b.certified_score
