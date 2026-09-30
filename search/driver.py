"""Experiment driver: multiprocessing rounds, checkpointing, resume, leaderboard.

    python -m search.driver configs/pilot_a26_bh.yaml

A run is a sequence of *rounds*. In round r worker w uses rng = default_rng([seed, w, r]) and the
method-specific state it left in the checkpoint, so a run is reproducible for fixed
(workers, iters_per_round, rounds) and resumable at any round boundary.
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Records (see docs/records.md): Packomania csqv (sum of radii, all n, 12 decimals, retrieved 2026-09-28)
# and Packomania csq (max-min distance, 12 decimals). The 6-digit AlphaEvolve numbers for n=26/32 are
# superseded by these (n=26: 2.635983084919, n=32: 2.939572771205).
ALPHAEVOLVE_A = {26: 2.635862, 32: 2.937944}  # first AlphaEvolve values (May 2025), for the paper only
TOL = 1e-11  # records are printed with 12 decimals


AC1_RECORD = 1.50287  # best published upper bound on C1 (TTT-Discover, 30000 pieces; arXiv:2601.16175 Table 2)
AC1_TOL = 5e-6  # the record is published with 5 decimals
AC1_HUMAN = 1.50973  # best human (51 pieces), same table


def load_record(problem: str, n: int) -> float | None:
    if problem == "autocorr1":
        return AC1_RECORD
    # Packomania tables are NOT part of the repository: run `python scripts/fetch_packomania.py` (or `make fetch-data`) to download them
    # into ./data. Without them there is simply no published reference (classify() returns "no-published-reference").
    key, name = ("sumradii", "packomania_csqv_records.json") if problem == "sum_radii" else ("distance", "packomania_csq_records.json")
    f = ROOT / "data" / name
    if not f.exists():
        return None
    d = json.load(open(f))[key]
    return float(d[str(n)]) if str(n) in d else None


def classify(problem: str, n: int, score: float) -> str:
    """match: score >= REC - TOL (for sum_radii additionally allowing the documented n*1e-12 certificate
    shrink); improvement candidate: score > REC + TOL (strict, after any shrink)."""
    rec = load_record(problem, n)
    if rec is None:
        return "no-published-reference"
    if problem == "autocorr1":  # minimisation: smaller is better
        if score < rec - AC1_TOL:
            return "IMPROVEMENT-CANDIDATE(uncertified)"
        if score <= rec + AC1_TOL:
            return "match"
        if score <= 1.50317 + AC1_TOL:
            return "above-record(between AlphaEvolve-V2 and best)"
        if score < 1.50530 - AC1_TOL:
            return "above-record(better than AlphaEvolve-v1 1.50530, not than V2)"
        if score <= 1.50530 + AC1_TOL:
            return "above-record(matches AlphaEvolve 1.50530)"
        return "above-record(beats-best-human)" if score < AC1_HUMAN else "above-record"
    slack = TOL + (n * 1e-12 if problem == "sum_radii" else 0.0)
    if score > rec + TOL:
        return "IMPROVEMENT-CANDIDATE(uncertified)"
    if score >= rec - slack:
        return "match"
    return "below"


# ------------------------------------------------------------------------------ workers
_SEED_CACHE: dict = {}  # per worker process: seed solutions already converted/optimised


def _work(args):
    from search.methods import METHODS
    from search.problems import make_problem

    kind, n, method, seed, w, r, iters, state = args[:8]
    extras = args[8] if len(args) > 8 else {}
    P = make_problem(kind, n)
    if extras.get("seed_files"):
        from search.methods import convert_to_n

        ckey = (n, tuple((f, os.path.getmtime(f) if os.path.exists(f) else 0) for f in extras["seed_files"]))
        if ckey in _SEED_CACHE:
            P.seed_zs = _SEED_CACHE[ckey]
            extras = {}
    if extras.get("seed_files"):
        zs = []
        rng0 = np.random.default_rng([seed, 999])
        for f in extras["seed_files"]:
            try:
                d = json.load(open(f))
                for cand in (d["candidates"] if "candidates" in d else [d]):
                    if "circles" not in cand:
                        continue
                    a = np.array([[float(v) for v in row] for row in cand["circles"]])
                    z0 = np.concatenate([a[:, 0], a[:, 1], a[:, 2]])
                    zc = convert_to_n(P, z0, len(a), rng0)
                    if zc is not None:
                        zs.append(P.local_opt(zc).z)
            except Exception:
                pass
        P.seed_zs = zs
        _SEED_CACHE[ckey] = zs
    rng = np.random.default_rng([seed, w, r])
    t0 = time.time()
    state, z, s, solves = METHODS[method](P, rng, iters, state)
    return {"w": w, "state": state, "z": z, "score": float(s), "solves": int(solves), "sec": time.time() - t0}


# ------------------------------------------------------------------------------ leaderboard
LB_FIELDS = ["problem", "n", "best_score", "record", "gap", "status", "method", "seed", "runtime_s",
              "certified_score", "updated"]


def read_leaderboard() -> dict[tuple[str, int], dict]:
    p = RESULTS / "leaderboard.csv"
    out: dict[tuple[str, int], dict] = {}
    if p.exists():
        for row in csv.DictReader(open(p, newline="")):
            out[(row["problem"], int(row["n"]))] = row
    return out


def write_leaderboard(rows: dict[tuple[str, int], dict]) -> None:
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "leaderboard.csv"
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=LB_FIELDS)
        wr.writeheader()
        for k in sorted(rows):
            wr.writerow({f: rows[k].get(f, "") for f in LB_FIELDS})
    os.replace(tmp, p)


def refresh_leaderboard() -> None:
    """Recompute record / gap / status columns from the current record tables."""
    rows = read_leaderboard()
    for (problem, n), r in rows.items():
        rec = load_record(problem, n)
        score = float(r["certified_score"]) if r.get("certified_score") else float(r["best_score"])
        r["record"] = "" if rec is None else f"{rec:.12f}"
        r["gap"] = "" if rec is None else f"{float(r['best_score']) - rec:+.3e}"
        r["status"] = classify(problem, n, score)
    write_leaderboard(rows)


def update_leaderboard(problem: str, n: int, score: float, method: str, seed: int, runtime: float, z, P) -> bool:
    """Record a candidate (float search score, NOT certified). Returns True if it is a new best."""
    rows = read_leaderboard()
    key = (problem, n)
    old = rows.get(key)
    if old and float(old["best_score"]) >= score:
        return False
    rec = load_record(problem, n)
    (RESULTS / "best").mkdir(parents=True, exist_ok=True)
    cand = P.to_certificate(z)
    cand["score_float"] = score
    cand["method"], cand["seed"] = method, seed
    json.dump(cand, open(RESULTS / "best" / f"{problem}_n{n}.json", "w"), indent=1)
    rows[key] = {"problem": problem, "n": n, "best_score": f"{score:.12f}",
                 "record": "" if rec is None else f"{rec:.12f}", "gap": "" if rec is None else f"{score - rec:+.3e}",
                 "status": classify(problem, n, score), "method": method, "seed": seed,
                 "runtime_s": f"{runtime:.0f}", "certified_score": (old or {}).get("certified_score", ""),
                 "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_leaderboard(rows)
    return True


# ------------------------------------------------------------------------------ experiment loop
def run_experiment(cfg: dict, pool=None, quiet: bool = False) -> dict:
    from search.methods import migrate
    from search.problems import make_problem

    kind, n, method = cfg["problem"], int(cfg["n"]), cfg["method"]
    name = cfg.get("name", f"{kind}_n{n}_{method}")
    seed = int(cfg.get("seed", 0))
    W = int(cfg.get("workers", os.cpu_count() or 4))
    iters = int(cfg.get("iters_per_round", 20))
    max_rounds = int(cfg.get("max_rounds", 10**9))
    tlimit = float(cfg.get("time_limit_s", 600))
    stop_at = cfg.get("stop_at")  # stop when the float score exceeds this
    log_to_lb = cfg.get("leaderboard", True)
    exp_dir = RESULTS / "runs" / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    ck = exp_dir / "ckpt.pkl"
    P = make_problem(kind, n)

    if ck.exists() and cfg.get("resume", True):
        c = pickle.load(open(ck, "rb"))
        rnd, states, best_s, best_z, elapsed, best_hist = c["round"], c["states"], c["best_s"], c["best_z"], c["elapsed"], c["hist"]
        if not quiet:
            print(f"[{name}] resuming at round {rnd}, best {best_s:.9f}, elapsed {elapsed:.0f}s", flush=True)
    else:
        rnd, states, best_s, best_z, elapsed, best_hist = 0, [None] * W, -np.inf, None, 0.0, []
        (exp_dir / "log.jsonl").write_text("")
    if len(states) != W:
        states = (states + [None] * W)[:W]

    own_pool = pool is None
    if own_pool:
        pool = mp.get_context("spawn").Pool(W)
    t_start = time.time()
    last_ck = time.time()
    try:
        while rnd < max_rounds and elapsed + (time.time() - t_start) < tlimit:
            args = [(kind, n, method, seed, w, rnd, iters, states[w], cfg.get("extras", {})) for w in range(W)]
            outs = pool.map(_work, args, chunksize=1)
            for o in outs:
                states[o["w"]] = o["state"]
            if method == "evo":
                states = migrate(states)
            top = max(outs, key=lambda o: o["score"])
            if top["score"] > best_s:
                best_s, best_z = top["score"], top["z"]
            rnd += 1
            now = elapsed + (time.time() - t_start)
            best_hist.append((now, best_s))
            row = {"round": rnd, "t": round(now, 1), "best": best_s, "round_best": top["score"],
                   "worker_scores": [round(o["score"], 6) for o in outs], "solves": sum(o["solves"] for o in outs)}
            with open(exp_dir / "log.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            if not quiet:
                print(f"[{name}] r{rnd} t={now:.0f}s best={best_s:.9f} round_best={top['score']:.9f}", flush=True)
            if log_to_lb and best_z is not None:
                update_leaderboard(kind, n, best_s, method, seed, now, best_z, P)
            if time.time() - last_ck > float(cfg.get("checkpoint_every_s", 120)) or True:
                tmp = ck.with_suffix(".tmp")
                pickle.dump({"round": rnd, "states": states, "best_s": best_s, "best_z": best_z,
                             "elapsed": now, "hist": best_hist}, open(tmp, "wb"))
                os.replace(tmp, ck)
                last_ck = time.time()
            if stop_at is not None and best_s > float(stop_at):
                break
    finally:
        if own_pool:
            pool.terminate()
            pool.join()
    summ = {"name": name, "problem": kind, "n": n, "method": method, "seed": seed, "workers": W,
            "rounds": rnd, "elapsed_s": elapsed + (time.time() - t_start), "best_score": best_s,
            "status": classify(kind, n, best_s), "record": load_record(kind, n), "hist": best_hist,
            "cfg": cfg}
    json.dump(summ, open(exp_dir / "summary.json", "w"), indent=1)
    if best_z is not None:
        json.dump(P.to_certificate(best_z) | {"score_float": best_s}, open(exp_dir / "best.json", "w"), indent=1)
    return summ


def main() -> None:
    cfg = yaml.safe_load(open(sys.argv[1]))
    if isinstance(cfg, list):
        for c in cfg:
            run_experiment(c)
    else:
        run_experiment(cfg)


if __name__ == "__main__":
    mp.freeze_support()
    main()
