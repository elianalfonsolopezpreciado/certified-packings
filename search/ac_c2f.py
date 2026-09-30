"""Problem C driver: homotopy from scratch, upsampling (coarse-to-fine), SLP polish, certification, progress log.

    python -m search.ac_c2f scratch  --N 600  --B 64 --stages 100 --steps 3000 --seed 1
    python -m search.ac_c2f upsample --src 600 --dst 1200 --noise 0.003 --stages 40 --steps 1000 --seed 1
    python -m search.ac_c2f certify  --N 600

Every candidate is stored as float heights in results/autocorr1/cand_N<k>.json (best only); `certify` turns the best
candidate into certificates/C_N<k>.json[.gz] and runs the independent verifier (both backends) in fresh processes.
`results/C_progress.csv` logs every improvement (float value) and every certification (certified exact value).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402

from search.ac_gpu import heights_from_theta, homotopy, theta_from_heights  # noqa: E402
from search.autocorr import ratio, slp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CAND = ROOT / "results" / "autocorr1"
PROG = ROOT / "results" / "C_progress.csv"
CERTS = ROOT / "certificates"
FIELDS = ["time", "N", "kind", "method", "seed", "ratio", "note"]
REC = {"alphaevolve": 1.50530, "alphaevolve_v2": 1.50317, "ttt_discover": 1.50287, "best_human": 1.50973}


def log(N: int, kind: str, method: str, seed, value: float, note: str = "") -> None:
    PROG.parent.mkdir(exist_ok=True)
    new = not PROG.exists()
    with open(PROG, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(FIELDS)
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), N, kind, method, seed, f"{value:.12f}", note])


def cand_path(N: int) -> Path:
    return CAND / f"cand_N{N}.json"


def load_cand(N: int):
    p = cand_path(N)
    if not p.exists():
        return None
    d = json.load(open(p))
    return np.array([float(v) for v in d["heights"]]), d["ratio_float"]


def save_cand(a: np.ndarray, method: str, seed, note: str = "") -> bool:
    """Keep only the best candidate per N. Returns True if it is a new best."""
    N = len(a)
    r = ratio(a)
    old = load_cand(N)
    if old is not None and old[1] <= r:
        return False
    CAND.mkdir(parents=True, exist_ok=True)
    a = a * (N / a.sum())
    json.dump({"problem": "autocorr1", "n": N, "heights": [repr(float(v)) for v in a], "ratio_float": ratio(a),
               "method": method, "seed": seed, "note": note}, open(cand_path(N), "w"))
    log(N, "float", method, seed, ratio(a), note)
    return True


def polish(a: np.ndarray, seconds: float) -> np.ndarray:
    """SLP with near-active rows: dense LP, only sensible for N <= ~2500."""
    if len(a) > 2500 or seconds <= 0:
        return a
    b, r, _ = slp(a, seconds)
    return b if r < ratio(a) else a


def cmd_scratch(a) -> None:
    rng = np.random.default_rng(a.seed)
    th0 = a.init_std * rng.standard_normal((a.B, a.N))
    t0 = time.time()
    th, r = homotopy(th0, a.tau0, a.tau1, a.stages, a.steps, lr=a.lr, lr_end=a.lr_end)
    order = np.argsort(r)[: a.top]
    print(f"[scratch N={a.N}] homotopy {time.time()-t0:.0f}s best={r.min():.6f} median={np.median(r):.6f}", flush=True)
    for k in order:
        h = heights_from_theta(th[k])
        save_cand(h, "gpu-homotopy", a.seed, f"B={a.B} stages={a.stages} steps={a.steps}")
        h2 = polish(h, a.slp_seconds)
        rr = ratio(h2)
        print(f"   cand {k}: homotopy {r[k]:.6f} -> polished {rr:.6f}", flush=True)
        save_cand(h2, "gpu-homotopy+slp", a.seed, f"B={a.B} stages={a.stages} steps={a.steps}")
    print("best float at N:", load_cand(a.N)[1])


def cmd_upsample(a) -> None:
    src = load_cand(a.src)
    if src is None:
        raise SystemExit(f"no candidate for N={a.src}")
    k = a.dst // a.src
    assert k * a.src == a.dst, "dst must be a multiple of src"
    base = np.repeat(src[0], k)
    rng = np.random.default_rng(a.seed)
    B = a.B
    th0 = np.stack([theta_from_heights(base * np.exp(a.noise * rng.standard_normal(a.dst) * (i > 0))) for i in range(B)])
    t0 = time.time()
    th, r = homotopy(th0, a.tau0, a.tau1, a.stages, a.steps, lr=a.lr, lr_end=a.lr_end)
    k_best = int(np.argmin(r))
    print(f"[upsample {a.src}->{a.dst}] {time.time()-t0:.0f}s start ratio {ratio(base):.9f}  best {r.min():.9f} (row {k_best})",
          flush=True)
    for i in np.argsort(r)[: a.top]:
        h = heights_from_theta(th[i])
        save_cand(h, "gpu-upsample", a.seed, f"{a.src}->{a.dst} noise={a.noise}")
        h2 = polish(h, a.slp_seconds)
        save_cand(h2, "gpu-upsample+slp", a.seed, f"{a.src}->{a.dst} noise={a.noise}")
    print("best float at N:", load_cand(a.dst)[1])


def cmd_multilevel(a) -> None:
    """Multilevel continuation: resolution doubles while tau decays; top rows are replicated with noise at each doubling."""
    rng = np.random.default_rng(a.seed)
    levels = [a.N0 * 2**k for k in range(a.levels)]
    L = len(levels)
    lt0, lt1 = np.log(a.tau0), np.log(a.tau1)
    th = a.init_std * rng.standard_normal((a.B, levels[0]))
    t0 = time.time()
    for li, N in enumerate(levels):
        ta = float(np.exp(lt0 + (lt1 - lt0) * li / L))
        tb = float(np.exp(lt0 + (lt1 - lt0) * (li + 1) / L))
        th, r = homotopy(th, ta, tb, a.stages, a.steps, lr=a.lr, lr_end=a.lr_end)
        order = np.argsort(r)
        print(f"[multilevel N={N}] tau {ta:.1e}->{tb:.1e} best={r.min():.6f} median={np.median(r):.6f} ({time.time()-t0:.0f}s)", flush=True)
        h = heights_from_theta(th[order[0]])
        save_cand(h, "gpu-multilevel", a.seed, f"N0={a.N0} levels={a.levels} B={a.B}")
        if li + 1 < L:
            top = th[order[: a.keep]]
            reps = int(np.ceil(a.B / a.keep))
            th = np.repeat(top, 2, axis=1)  # piecewise-constant refinement of log-heights
            th = np.tile(th, (reps, 1))[: a.B]
            th = th + a.noise * rng.standard_normal(th.shape) * (np.arange(a.B)[:, None] >= a.keep)
    order = np.argsort(r)
    for i in order[: a.top]:
        h = heights_from_theta(th[i])
        save_cand(h, "gpu-multilevel", a.seed, f"N0={a.N0} levels={a.levels}")
        h2 = polish(h, a.slp_seconds)
        save_cand(h2, "gpu-multilevel+slp", a.seed, f"N0={a.N0} levels={a.levels}")
        print(f"   final row {i}: {r[i]:.6f} -> polished {ratio(h2):.6f}", flush=True)
    print("best float at N=%d:" % levels[-1], load_cand(levels[-1])[1])


def run_verifier(path: Path, backend: str, out: Path) -> dict:
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, "-m", "verify", str(path), "--backend", backend, "--out", str(out)], cwd=ROOT,
                       capture_output=True, text=True, env=env)
    rep = json.load(open(out)) if out.exists() else {"all_valid": False, "reports": []}
    rep["returncode"] = r.returncode
    return rep


def cmd_certify(a) -> None:
    got = load_cand(a.N)
    if got is None:
        raise SystemExit("no candidate")
    h, rf = got
    CERTS.mkdir(exist_ok=True)
    name = f"C_N{a.N}"
    big = a.N > 5000
    cpath = CERTS / (name + (".json.gz" if big else ".json"))
    cert = {"problem": "autocorr1", "n": a.N, "heights": [repr(float(v)) for v in h],
            "provenance": {"candidate": str(cand_path(a.N).relative_to(ROOT)), "float_ratio": rf}}
    # do not overwrite a better certificate
    prev = CERTS / f"{name}.summary.json"
    if prev.exists():
        pv = json.load(open(prev))
        if pv.get("ok") and float(pv["certified_score_fraction"]) <= ratio(h) + 1e-15:
            print("existing certificate is at least as good:", pv["certified_score_fraction"])
            return
    if big:
        with gzip.open(cpath, "wt") as fh:
            json.dump(cert, fh)
    else:
        json.dump(cert, open(cpath, "w"), indent=0)
    t0 = time.time()
    rep_fr = run_verifier(cpath, "fraction", CERTS / f"{name}.verify_fraction.json")
    rep_mp = run_verifier(cpath, "mpmath", CERTS / f"{name}.verify_mpmath.json")
    ok = bool(rep_fr["all_valid"] and rep_mp["all_valid"])
    out = {"name": name, "N": a.N, "ok": ok, "verify_seconds": round(time.time() - t0, 1)}
    if ok:
        s_fr = rep_fr["reports"][0]["certified_score"]
        s_mp = rep_mp["reports"][0]["certified_score"]
        out.update(certified_score_fraction=s_fr, certified_score_mpmath=s_mp, backends_equal=(s_fr == s_mp))
        log(a.N, "certified", "verify(kronecker|limbs)", "", float(s_fr), "exact; both backends " + ("agree" if s_fr == s_mp else "DISAGREE"))
    json.dump(out, open(prev, "w"), indent=1)
    print(json.dumps(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scratch", "upsample"):
        s = sub.add_parser(name)
        s.add_argument("--B", type=int, default=32)
        s.add_argument("--stages", type=int, default=40)
        s.add_argument("--steps", type=int, default=1000)
        s.add_argument("--tau0", type=float, default=3e-2 if name == "scratch" else 1e-4)
        s.add_argument("--tau1", type=float, default=1e-6)
        s.add_argument("--lr", type=float, default=0.03 if name == "scratch" else 0.003)
        s.add_argument("--lr-end", type=float, default=0.003 if name == "scratch" else 0.0003)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--top", type=int, default=4)
        s.add_argument("--slp-seconds", type=float, default=120)
        if name == "scratch":
            s.add_argument("--N", type=int, required=True)
            s.add_argument("--init-std", type=float, default=0.3)
        else:
            s.add_argument("--src", type=int, required=True)
            s.add_argument("--dst", type=int, required=True)
            s.add_argument("--noise", type=float, default=0.003)
    m = sub.add_parser("multilevel")
    m.add_argument("--N0", type=int, default=50)
    m.add_argument("--levels", type=int, default=5)
    m.add_argument("--B", type=int, default=64)
    m.add_argument("--keep", type=int, default=8)
    m.add_argument("--stages", type=int, default=20)
    m.add_argument("--steps", type=int, default=800)
    m.add_argument("--tau0", type=float, default=3e-2)
    m.add_argument("--tau1", type=float, default=1e-6)
    m.add_argument("--lr", type=float, default=0.03)
    m.add_argument("--lr-end", type=float, default=0.003)
    m.add_argument("--noise", type=float, default=0.02)
    m.add_argument("--init-std", type=float, default=0.3)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--top", type=int, default=3)
    m.add_argument("--slp-seconds", type=float, default=120)
    c = sub.add_parser("certify")
    c.add_argument("--N", type=int, required=True)
    a = ap.parse_args()
    {"scratch": cmd_scratch, "upsample": cmd_upsample, "certify": cmd_certify, "multilevel": cmd_multilevel}[a.cmd](a)


if __name__ == "__main__":
    main()
