"""Phase-5 certification pipeline.

    python -m search.certify sum_radii 26 [candidate.json]

1. load float candidate (default results/best/<problem>_n<n>.json)
2. high-precision active-set Newton polish (search.polish)
3. write certificates/<problem>_n<n>.json
4. run the INDEPENDENT verifier in fresh processes: mpmath backend and exact-rational backend,
   store both outputs next to the certificate
5. only if both pass: record certified score in results/leaderboard.csv; classify vs the record.

Nothing is claimed unless step 4 passes for both backends.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402

from search.driver import ROOT, classify, load_record, read_leaderboard, write_leaderboard  # noqa: E402
from search.polish import polish, polish_mixed  # noqa: E402
from search.problems import make_problem  # noqa: E402

CERT_DIR = ROOT / "certificates"


def _z_from_candidate(kind: str, n: int, cand: dict) -> np.ndarray:
    if kind == "sum_radii":
        c = np.array([[float(v) for v in row] for row in cand["circles"]])
        return np.concatenate([c[:, 0], c[:, 1], c[:, 2]])
    p = np.array([[float(v) for v in row] for row in cand["points"]])
    z = np.concatenate([p[:, 0], p[:, 1], [0.0]])
    return z


def run_verifier(path: Path, backend: str, out: Path) -> dict:
    """Fresh-process call of the independent verifier."""
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, "-m", "verify", str(path), "--backend", backend, "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    rep = json.load(open(out)) if out.exists() else {"all_valid": False, "reports": []}
    rep["returncode"] = r.returncode
    rep["stdout"] = r.stdout
    return rep


def certify_autocorr(n: int) -> dict:
    """Problem C: the float heights (decimal text) ARE the certificate; the verifier computes the exact ratio."""
    CERT_DIR.mkdir(exist_ok=True)
    src = ROOT / "results" / "autocorr1" / f"best_N{n}.json"
    cand = json.load(open(src))
    cert = {"problem": "autocorr1", "n": n, "heights": cand["heights"],
            "provenance": {"candidate": str(src.relative_to(ROOT)), "method": "gpu-pnorm+slp", "seed": cand.get("seed")}}
    name = f"autocorr1_n{n}"
    cpath = CERT_DIR / f"{name}.json"
    json.dump(cert, open(cpath, "w"), indent=1)
    rep_mp = run_verifier(cpath, "mpmath", CERT_DIR / f"{name}.verify_mpmath.json")
    rep_fr = run_verifier(cpath, "fraction", CERT_DIR / f"{name}.verify_fraction.json")
    ok = bool(rep_mp["all_valid"] and rep_fr["all_valid"])
    out = {"name": name, "ok": ok}
    if ok:
        s_fr = rep_fr["reports"][0]["certified_score"]
        out.update(certified_score_fraction=s_fr, certified_score_mpmath=rep_mp["reports"][0]["certified_score"],
                   record=load_record("autocorr1", n), status=classify("autocorr1", n, float(s_fr)))
        rows = read_leaderboard()
        key = ("autocorr1", n)
        score = float(s_fr)
        if key not in rows or score < float(rows[key]["best_score"]):  # minimisation problem
            rows[key] = {"problem": "autocorr1", "n": n, "best_score": f"{score:.12f}",
                         "record": f"{load_record('autocorr1', n):.12f}", "gap": f"{score - load_record('autocorr1', n):+.3e}",
                         "status": classify("autocorr1", n, score), "method": "gpu-pnorm+slp", "seed": cand.get("seed", ""),
                         "runtime_s": "", "certified_score": f"{score:.12f}", "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
            write_leaderboard(rows)
    json.dump(out, open(CERT_DIR / f"{name}.summary.json", "w"), indent=1)
    return out


def certify(kind: str, n: int, cand_path: str | None = None, tag: str | None = None) -> dict:
    if kind == "autocorr1":
        return certify_autocorr(n)
    CERT_DIR.mkdir(exist_ok=True)
    src = Path(cand_path) if cand_path else ROOT / "results" / "best" / f"{kind}_n{n}.json"
    cand = json.load(open(src))
    P = make_problem(kind, n)
    z = _z_from_candidate(kind, n, cand)
    loc = P.local_opt(z, maxiter=500)  # re-tighten to a proper local optimum first
    if loc.score >= (P.score_raw(z) - 1e-9):
        z = loc.z
    pol = polish_mixed(n, z) if (kind == "sum_radii" and n >= 60) else polish(kind, n, z)
    cert = pol["certificate"]
    cert["provenance"] = {"candidate": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
                          "method": cand.get("method"), "seed": cand.get("seed"), "polish": pol["info"]}
    name = (f"A_n{n}" if (kind == "sum_radii" and n >= 100) else f"{kind}_n{n}") + (f"_{tag}" if tag else "")
    cpath = CERT_DIR / f"{name}.json"
    json.dump(cert, open(cpath, "w"), indent=1)
    rep_mp = run_verifier(cpath, "mpmath", CERT_DIR / f"{name}.verify_mpmath.json")
    rep_fr = run_verifier(cpath, "fraction", CERT_DIR / f"{name}.verify_fraction.json")
    ok = bool(rep_mp["all_valid"] and rep_fr["all_valid"])
    out = {"name": name, "ok": ok, "polish": pol["info"]}
    if ok:
        s_mp = rep_mp["reports"][0]["certified_score"]
        s_fr = rep_fr["reports"][0]["certified_score"]
        out["certified_score_mpmath"], out["certified_score_fraction"] = s_mp, s_fr
        out["agree_1e-20"] = abs(float(s_mp) - float(s_fr)) < 1e-20
        score = float(s_fr)
        out["record"] = load_record(kind, n)
        out["status"] = classify(kind, n, score)
        rows = read_leaderboard()
        key = (kind, n)
        if key in rows and (rows[key].get("certified_score", "") == "" or float(rows[key]["certified_score"]) < score):
            rows[key]["certified_score"] = f"{score:.12f}"
            rows[key]["status"] = classify(kind, n, score)
            write_leaderboard(rows)
    json.dump(out, open(CERT_DIR / f"{name}.summary.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    kind, n = sys.argv[1], int(sys.argv[2])
    print(json.dumps(certify(kind, n, sys.argv[3] if len(sys.argv) > 3 else None), indent=1))
