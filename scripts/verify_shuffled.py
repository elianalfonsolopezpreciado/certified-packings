"""Re-verify every certificate in several rounds with a different process order and a different PYTHONHASHSEED each round.

    python scripts/verify_shuffled.py [--rounds 3] [--backend both]

Each certificate is checked by a fresh `python -m verify` process; the certified scores are collected per round and must be identical across rounds and
identical to the scores stored in certificates/*.verify_{fraction,mpmath}.json. Exit code 0 iff everything agrees. (The verifier is deterministic; this guards against
accidental dependence on process order, hash randomisation or leftover state.)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def certificates() -> list[Path]:
    out = []
    for p in sorted((ROOT / "certificates").iterdir()):
        n = p.name
        if any(t in n for t in (".verify_", ".summary", ".eps")):
            continue
        if n.endswith(".json") or n.endswith(".json.gz"):
            out.append(p)
    return out


def run_one(path: Path, hashseed: int) -> dict:
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed), PYTHONPATH=str(ROOT), MPMATH_NOGMPY="1")
    tmp = ROOT / "results" / f"_shuffle_{os.getpid()}.json"
    r = subprocess.run([sys.executable, "-m", "verify", str(path), "--backend", "both", "--out", str(tmp)], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    rep = json.load(open(tmp)) if tmp.exists() else {"all_valid": False, "reports": []}
    tmp.unlink(missing_ok=True)
    return {"rc": r.returncode, "valid": rep["all_valid"], "scores": [x["certified_score"] for x in rep["reports"]]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    certs = certificates()
    ref: dict[str, list[str]] = {}
    bad = 0
    for rnd in range(1, a.rounds + 1):
        order = certs[:]
        random.Random(rnd).shuffle(order)
        for p in order:
            res = run_one(p, hashseed=rnd)
            if not res["valid"] or res["rc"] != 0:
                print(f"round {rnd}: {p.name} INVALID")
                bad += 1
            if p.name in ref and ref[p.name] != res["scores"]:
                print(f"round {rnd}: {p.name} scores differ from an earlier round")
                bad += 1
            ref.setdefault(p.name, res["scores"])
        print(f"round {rnd} (order seed {rnd}, PYTHONHASHSEED {rnd}): {len(order)} certificates checked", flush=True)
    # compare with the stored verifier outputs written at certification time
    for p in certs:
        stem = p.name.replace(".json.gz", "").replace(".json", "")
        for backend, idx in (("fraction", 0), ("mpmath", 1)):
            f = ROOT / "certificates" / f"{stem}.verify_{backend}.json"
            if not f.exists():
                continue
            stored = json.load(open(f))["reports"][0]["certified_score"]
            if ref[p.name][idx] != stored:
                print(f"{p.name}: {backend} score {ref[p.name][idx]} != stored {stored}")
                bad += 1
    print("ALL CONSISTENT" if bad == 0 else f"{bad} PROBLEMS")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
