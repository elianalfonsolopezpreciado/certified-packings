"""Tests for the stand-alone .pck validator (validator/validate_pck.py): it accepts our files and rejects tampered copies."""
from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validate_pck", ROOT / "validator" / "validate_pck.py")
vp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vp)  # the validator itself imports nothing from this repository


def _rows(path: Path):
    lines = path.read_text().split("\n")
    return lines[0], lines[1], [ln.split() for ln in lines[2:] if ln.strip()]


def _write(path: Path, first, author, rows):
    path.write_text("\n".join([first, author] + [" ".join(r) for r in rows]) + "\n")


def _shift_toward_neighbour(rows, k, delta: Decimal):
    """Move circle k by `delta` straight towards its nearest circle (creates an overlap of about delta - slack)."""
    x, y = Decimal(rows[k][0]), Decimal(rows[k][1])
    best, bj = None, None
    for j, r in enumerate(rows):
        if j != k:
            d = ((Decimal(r[0]) - x) ** 2 + (Decimal(r[1]) - y) ** 2).sqrt()
            if best is None or d < best:
                best, bj = d, j
    ux, uy = (Decimal(rows[bj][0]) - x) / best, (Decimal(rows[bj][1]) - y) / best
    new = [f"{x + ux * delta:.20f}", f"{y + uy * delta:.20f}", rows[k][2]]
    return new


@pytest.mark.parametrize("n", [120, 250])
def test_submission_files_pass_with_zero_tolerance_and_3n_contacts(n):
    rep = vp.validate(str(ROOT / "submission" / f"csqv{n}.pck"), require_contacts=True)
    assert rep["valid"], rep["problems"]
    assert rep["n"] == n and rep["contacts_within_tol"] >= 3 * n
    assert len(str(rep["sum_radii"]).replace(".", "")) >= 15


@pytest.mark.parametrize("n", [120, 250])
def test_one_circle_shifted_by_1e9_is_rejected(n, tmp_path):
    first, author, rows = _rows(ROOT / "submission" / f"csqv{n}.pck")
    rows[len(rows) // 2] = _shift_toward_neighbour(rows, len(rows) // 2, Decimal("1e-9"))
    p = tmp_path / "shifted.pck"
    _write(p, first, author, rows)
    rep = vp.validate(str(p))
    assert not rep["valid"] and any("overlap" in m or "leaves the square" in m for m in rep["problems"])


@pytest.mark.parametrize("n", [120, 250])
def test_one_radius_plus_1e9_is_rejected(n, tmp_path):
    first, author, rows = _rows(ROOT / "submission" / f"csqv{n}.pck")
    k = len(rows) // 3
    rows[k][2] = f"{Decimal(rows[k][2]) + Decimal('1e-9'):.20f}"
    p = tmp_path / "bigger.pck"
    _write(p, first, author, rows)
    # a larger radius may also break the sort order / line-1 check; the point is that the packing is rejected
    assert not vp.validate(str(p))["valid"]


def test_tiny_perturbations_below_the_1e12_slack_are_still_accepted(tmp_path):
    """Sanity: the validator is exact, not paranoid - a shift far below our 1e-12 safety slack keeps the file valid."""
    first, author, rows = _rows(ROOT / "submission" / "csqv120.pck")
    rows[10] = _shift_toward_neighbour(rows, 10, Decimal("1e-14"))
    p = tmp_path / "tiny.pck"
    _write(p, first, author, rows)
    assert vp.validate(str(p))["valid"]


def test_synthetic_cases(tmp_path):
    def mk(name, text):
        p = tmp_path / name
        p.write_text(text)
        return str(p)

    # two tangent circles: valid with ZERO tolerance (exactly touching), sorted, line 1 = largest
    assert vp.validate(mk("tangent.pck", "0.25\nx\n-0.25 0 0.25\n0.25 0 0.25\n"))["valid"]
    # overlap by 1e-12
    assert not vp.validate(mk("ov.pck", "0.25\nx\n-0.2499999999995 0 0.25\n0.2499999999995 0 0.25\n"))["valid"]
    # outside the square
    assert not vp.validate(mk("out.pck", "0.6\nx\n0 0 0.6\n"))["valid"]
    # negative radius
    assert not vp.validate(mk("neg.pck", "0.1\nx\n0 0 -0.1\n"))["valid"]
    # first line is not the largest radius / not sorted
    assert not vp.validate(mk("l1.pck", "0.1\nx\n-0.25 0 0.25\n")) ["valid"]
    assert not vp.validate(mk("unsorted.pck", "0.25\nx\n-0.25 0 0.25\n0.4 0.4 0.05\n"))["valid"]
    # garbage / NaN
    for bad in ("0.25\nx\n0 0 nan\n", "0.25\nx\n0 0\n", "abc\nx\n0 0 0.1\n"):
        p = mk("bad.pck", bad)
        assert vp.main([p]) == 1


def test_validator_has_no_repository_imports():
    import ast

    tree = ast.parse((ROOT / "validator" / "validate_pck.py").read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"argparse", "sys", "decimal", "fractions", "__future__"}, mods


def test_cli_exit_codes(tmp_path):
    good = subprocess.run([sys.executable, str(ROOT / "validator" / "validate_pck.py"), str(ROOT / "submission" / "csqv120.pck")],
                          capture_output=True, text=True)
    assert good.returncode == 0 and "valid=True" in good.stdout
    first, author, rows = _rows(ROOT / "submission" / "csqv120.pck")
    rows[5][2] = f"{Decimal(rows[5][2]) + Decimal('1e-9'):.20f}"
    p = tmp_path / "t.pck"
    _write(p, first, author, rows)
    bad = subprocess.run([sys.executable, str(ROOT / "validator" / "validate_pck.py"), str(p)], capture_output=True, text=True)
    assert bad.returncode == 1 and "valid=False" in bad.stdout
