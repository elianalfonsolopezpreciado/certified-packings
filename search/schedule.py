"""Phase-4 scheduler.

    python -m search.schedule --problem sum_radii --ns 27 28 29 --method basin_hopping \
        --chunk 300 --max-chunks 6 --stall 3 --seed 11

For each n the run is executed in `chunk`-second pilots that RESUME the same experiment (same state,
time budget extended). Stop conditions per n (first one wins):
  * a strict improvement over the record (candidate) -> stop and hand over to search.certify,
  * `stall` consecutive chunks without progress (> 1e-10) -> switch to the next n,
  * `max_chunks` chunks used (budget).
Every chunk is appended to results/schedule_log.jsonl and the leaderboard is updated by the driver.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from search.driver import RESULTS, classify, load_record, run_experiment  # noqa: E402

ITERS = {"basin_hopping": 25, "evo": 25, "cmaes": 4000, "de": 8, "multistart": 10}


def run_n(problem: str, n: int, method: str, chunk: int, max_chunks: int, stall: int, seed: int, workers: int,
          tag: str = "sched", iters: int | None = None, seed_files: list[str] | None = None,
          keep_going: bool = False) -> dict:
    name = f"{tag}_{problem}_n{n}_{method}_s{seed}"
    best_prev, stalled, best = -1e9, 0, -1e9
    status = "budget"
    for c in range(1, max_chunks + 1):
        cfg = dict(name=name, problem=problem, n=n, method=method, seed=seed, workers=workers,
                   iters_per_round=iters or ITERS[method], time_limit_s=chunk * c, resume=True,
                   extras={"seed_files": seed_files or []})
        t0 = time.time()
        s = run_experiment(cfg, quiet=True)
        best = s["best_score"]
        progressed = best > best_prev + 1e-10
        stalled = 0 if progressed else stalled + 1
        st = classify(problem, n, best)
        row = {"name": name, "n": n, "method": method, "chunk": c, "wall_s": round(time.time() - t0),
               "best": best, "record": load_record(problem, n), "status": st, "stalled": stalled,
               "t": time.strftime("%H:%M:%S")}
        with open(RESULTS / "schedule_log.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        best_prev = max(best_prev, best)
        if st.startswith("IMPROVEMENT") and not keep_going:
            status = "improvement-candidate"
            break
        if stalled >= stall:
            status = "stalled"
            break
    return {"n": n, "method": method, "best": best, "status": status, "name": name}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="sum_radii")
    ap.add_argument("--ns", type=int, nargs="+", required=True)
    ap.add_argument("--method", default="basin_hopping")
    ap.add_argument("--chunk", type=int, default=300)
    ap.add_argument("--max-chunks", type=int, default=6)
    ap.add_argument("--stall", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--tag", default="sched")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--keep-going", action="store_true", help="do not stop when the float score exceeds the record")
    ap.add_argument("--seed-files", nargs="*", default=None,
                    help="candidate JSONs (sum_radii circles) of size n-1, n or n+1 used as chain starts; {n} is replaced by n")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    out = []
    for n in a.ns:
        sf = [f.replace("{n}", str(n)).replace("{n-1}", str(n - 1)).replace("{n+1}", str(n + 1)) for f in (a.seed_files or [])]
        out.append(run_n(a.problem, n, a.method, a.chunk, a.max_chunks, a.stall, a.seed, a.workers, a.tag, a.iters, sf, a.keep_going))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    mp.freeze_support()
    main()
