"""Reporting: ablation table, convergence curves, packing pictures, headline table.

    python -m search.report headline     # verify every certificates/*.json and rebuild the table
    python -m search.report ablation     # results/ablation.csv + convergence plots
    python -m search.report figures      # packing pictures for every certificate

The headline table is computed ONLY from certificates/*.json re-verified by the independent
`verify` package (nothing is read from the search logs).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPMATH_NOGMPY", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CERTS = ROOT / "certificates"


# ------------------------------------------------------------------------------ headline
def headline() -> str:
    from search.driver import classify, load_record
    from verify import load_certificate, parse, verify_fraction, verify_mpmath

    rows = []
    for p in sorted(list(CERTS.glob("*.json")) + list(CERTS.glob("*.json.gz"))):
        if any(s in p.name for s in (".verify_", ".summary", ".eps")):
            continue
        parsed = parse(load_certificate(str(p)))
        a, b = verify_fraction(parsed), verify_mpmath(parsed)
        ok = a.valid and b.valid
        kind, n = parsed.problem, parsed.n
        score = float(a.certified_score)
        rec = load_record(kind, n)
        if p.name.endswith(".gz"):
            import gzip

            prov = json.load(gzip.open(p, "rt", encoding="utf-8")).get("provenance", {})
        else:
            prov = json.load(open(p, encoding="utf-8")).get("provenance", {})
        rows.append({"file": p.name, "problem": kind, "n": n, "ok": ok, "score": a.certified_score[:22],
                     "agree": a.certified_score[:20] == b.certified_score[:20],
                     "record": "" if rec is None else f"{rec:.12g}",
                     "gap": "" if rec is None else f"{score - rec:+.2e}",
                     "raw": a.raw_score[:22],
                     "gap_raw": "" if rec is None else f"{float(a.raw_score) - rec:+.2e}",
                     "status": (classify(kind, n, score) if ok else "INVALID").replace("IMPROVEMENT-CANDIDATE(uncertified)", "IMPROVEMENT (certified, vs table of last record check)"),
                     "method": prov.get("method", "")})
    rows.sort(key=lambda r: (r["problem"], r["n"], r["file"]))
    md = ["| problem | n | certified score (exact/mpmath agree) | raw score (before shrink) | record | gap (certified) | gap (raw) | status | method | file |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['problem']} | {r['n']} | {r['score']} {'(agree)' if r['agree'] else '(DISAGREE)'} | {r['raw']} | "
                  f"{r['record']} | {r['gap']} | {r['gap_raw']} | {r['status']} | {r['method']} | {r['file']} |")
    text = "\n".join(md) + "\n"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "headline_table.md").write_text(text, encoding="utf-8")
    with open(RESULTS / "headline_table.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["file"])
        wr.writeheader()
        wr.writerows(rows)
    print(text)
    return text


# ------------------------------------------------------------------------------ ablation
def _runs():
    for d in sorted((RESULTS / "runs").glob("*")):
        s = d / "summary.json"
        if s.exists():
            yield d, json.load(open(s))


def ablation() -> None:
    if not (ROOT / "data" / "packomania_csqv_records.json").exists():
        print("WARNING: no local Packomania tables (data/ is not redistributed): the gap_to_record column will be BLANK. "
              "Run `python scripts/fetch_packomania.py` first if you want it.", file=sys.stderr)
    rows = []
    for d, s in _runs():
        solves = 0
        lg = d / "log.jsonl"
        if lg.exists():
            for line in open(lg):
                try:
                    solves += int(json.loads(line).get("solves", 0))
                except Exception:
                    pass
        hist = s.get("hist", [])
        if not hist and lg.exists():  # GPU stage logs: (t, polished_best)
            for line in open(lg):
                try:
                    r_ = json.loads(line)
                    if "polished_best" in r_:
                        hist.append((r_["t"], r_["polished_best"]))
                except Exception:
                    pass
            s["hist"] = hist
        final = s["best_score"]
        t_hit = ""
        for t, b in hist:
            if b >= final - 1e-9:
                t_hit = round(t)
                break
        from search.driver import load_record

        rec = load_record(s["problem"], s["n"])  # always the current record table, not the logged one
        rows.append({"run": s["name"], "problem": s["problem"], "n": s["n"], "method": s["method"],
                     "workers": s.get("workers", ""), "elapsed_s": round(s["elapsed_s"]), "best_score": f"{final:.9f}",
                     "gap_to_record": "" if rec is None else f"{final - rec:+.3e}", "time_to_best_s": t_hit,
                     "local_solves": solves or ""})
    RESULTS.mkdir(exist_ok=True)
    if rows:
        with open(RESULTS / "ablation.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        # markdown version
        keys = list(rows[0].keys())
        md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        md += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
        (RESULTS / "ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print("\n".join(md))
    # convergence plots per (problem, n)
    (RESULTS / "plots").mkdir(exist_ok=True)
    groups: dict[tuple[str, int], list] = {}
    for d, s in _runs():
        if s.get("hist"):
            groups.setdefault((s["problem"], s["n"]), []).append(s)
    for (kind, n), ss in groups.items():
        if len(ss) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for s in ss:
            t, b = zip(*s["hist"])
            ax.step(t, b, where="post", label=s["method"])
        rec = ss[0].get("record")
        if rec:
            ax.axhline(rec, color="k", ls="--", lw=0.8, label="record")
        top = max(s["best_score"] for s in ss)
        ax.set_ylim(top - (0.02 if kind == "sum_radii" else 0.01), top + 0.001)
        ax.set_xlabel("wall-clock (s, 12 workers)")
        ax.set_ylabel("best score")
        ax.set_title(f"{kind} n={n}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(RESULTS / "plots" / f"convergence_{kind}_n{n}.png", dpi=130)
        plt.close(fig)


# ------------------------------------------------------------------------------ latex tables
def _tex(x: object) -> str:
    return str(x).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def latex() -> None:
    """paper/tables_headline.tex and paper/tables_ablation.tex, generated from results/*.csv only."""
    out = ROOT / "paper"
    out.mkdir(exist_ok=True)
    nl = chr(10)
    bs = chr(92)  # backslash
    hp = RESULTS / "headline_table.csv"
    if hp.exists():
        rows = list(csv.DictReader(open(hp, newline="")))
        keys = ("problem", "n", "score", "record", "gap", "gap_raw", "status")
        L = [bs + "begin{longtable}{lllllll}", bs + "hline", " & ".join(_tex(k) for k in keys) + " " + bs + bs,
             bs + "hline", bs + "endhead"]
        for r in rows:
            r = dict(r, score=r["score"][:17], status=r["status"].replace("above-record", "above").replace("IMPROVEMENT (certified, vs table of last record check)", "IMPROVEMENT (certified)")[:34])
            L.append(" & ".join(_tex(r[k]) for k in keys) + " " + bs + bs)
        L += [bs + "hline", bs + "end{longtable}"]
        (out / "tables_headline.tex").write_text(nl.join(L) + nl, encoding="utf-8")
    ap_ = RESULTS / "ablation.csv"
    if ap_.exists():
        rows = list(csv.DictReader(open(ap_, newline="")))
        rows = [r for r in rows if r["run"].startswith(("pilot_", "pilotL_", "abl_", "baseline_"))]
        keys = ("run", "elapsed_s", "best_score", "gap_to_record", "time_to_best_s")
        for r in rows:
            r["run"] = (r["run"].replace("min_distance", "B").replace("sum_radii", "A").replace("basin_hopping", "bh")
                        .replace("multistart", "ms").replace("baseline_", "base_").replace("pilotL_", "").replace("pilot_", ""))
        L = [bs + "begin{tabular}{lllll}", bs + "hline", " & ".join(_tex(k) for k in keys) + " " + bs + bs,
             bs + "hline"]
        for r in rows:
            L.append(" & ".join(_tex(r[k]) for k in keys) + " " + bs + bs)
        L += [bs + "hline", bs + "end{tabular}"]
        (out / "tables_ablation.tex").write_text(nl.join(L) + nl, encoding="utf-8")


def sync_leaderboard_c() -> None:
    """Upsert one leaderboard row per certified Problem-C N (best certified value over all certificates with that N)."""
    import gzip
    import time

    from search.driver import classify, load_record, read_leaderboard, write_leaderboard

    rows = read_leaderboard()
    for p in sorted(list(CERTS.glob("*.json")) + list(CERTS.glob("*.json.gz"))):
        if any(t in p.name for t in (".verify_", ".summary", ".eps")):
            continue
        parsed = parse_cert(p)
        if parsed is None or parsed[0] != "autocorr1":
            continue
        _, n, score = parsed
        key = ("autocorr1", n)
        if key in rows and float(rows[key]["certified_score"]) <= score:
            continue
        rec = load_record("autocorr1", n)
        rows[key] = {"problem": "autocorr1", "n": n, "best_score": f"{score:.12f}", "record": f"{rec:.12f}",
                     "gap": f"{score - rec:+.3e}", "status": classify("autocorr1", n, score), "method": "gpu homotopy/melt",
                     "seed": "", "runtime_s": "", "certified_score": f"{score:.12f}", "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_leaderboard(rows)


def parse_cert(p):
    from verify import load_certificate, parse, verify_fraction

    try:
        parsed = parse(load_certificate(str(p)))
        rep = verify_fraction(parsed)
        return parsed.problem, parsed.n, float(rep.certified_score)
    except Exception:
        return None


def cplot() -> None:
    """results/plots/C_bound_vs_N.png from results/C_progress.csv (certified rows) with the published bounds."""
    rows = list(csv.DictReader(open(RESULTS / "C_progress.csv", newline="")))
    best: dict[int, float] = {}
    for r in rows:
        if r["kind"] == "certified":
            n, v = int(r["N"]), float(r["ratio"])
            best[n] = min(v, best.get(n, 9.0))
    ns = sorted(best)
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.semilogx(ns, [best[n] for n in ns], "o-", label="this work (certified)")
    for name, v, ls in (("best human 1.50973", 1.50973, ":"), ("AlphaEvolve 1.50530", 1.50530, "--"),
                        ("AlphaEvolve V2 1.50317", 1.50317, "-."), ("TTT-Discover 1.50287", 1.50287, "-")):
        ax.axhline(v, color="gray", ls=ls, lw=0.9, label=name)
    ax.set_xlabel("number of pieces N")
    ax.set_ylabel("certified upper bound R (smaller is better)")
    ax.set_ylim(1.5, 1.52)
    ax.legend(fontsize=7)
    fig.tight_layout()
    (RESULTS / "plots").mkdir(exist_ok=True)
    fig.savefig(RESULTS / "plots" / "C_bound_vs_N.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------ figures
def draw_autocorr(c: dict, out: Path) -> None:
    a = np.array([float(v) for v in c["heights"]])
    N = len(a)
    x = np.linspace(-0.25, 0.25, N + 1)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    axs[0].stairs(a / a.mean(), x, fill=True, alpha=0.5)
    axs[0].set_title(f"step function, N={N}", fontsize=9)
    conv = np.convolve(a, a) * (1.0 / (2 * N))
    t = np.linspace(-0.5, 0.5, 2 * N + 1)[1:-1]
    axs[1].plot(t, conv / (a.sum() / (2 * N)) ** 2, lw=0.8)
    R = 2 * N * np.convolve(a, a).max() / a.sum() ** 2
    axs[1].axhline(R, color="r", lw=0.6)
    axs[1].set_title(f"(f*f)/(int f)^2, max = {R:.6f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def draw(cert_path: Path, out: Path) -> None:
    c = json.load(open(cert_path))
    if c["problem"] == "autocorr1":
        return draw_autocorr(c, out)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=1.2))
    if c["problem"] == "sum_radii":
        for x, y, r in c["circles"]:
            ax.add_patch(plt.Circle((float(x), float(y)), float(r), fill=True, alpha=0.35, ec="k", lw=0.6))
        title = f"sum of radii, n={c['n']}"
    else:
        P = np.array([[float(a), float(b)] for a, b in c["points"]])
        ax.plot(P[:, 0], P[:, 1], "o", ms=4)
        d = min(np.hypot(*(P[i] - P[j])) for i in range(len(P)) for j in range(i + 1, len(P)))
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                if np.hypot(*(P[i] - P[j])) < d + 1e-9:
                    ax.plot(*zip(P[i], P[j]), "r-", lw=0.7)
        title = f"max-min distance, n={c['n']}"
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def figures() -> None:
    (RESULTS / "plots").mkdir(exist_ok=True)
    for p in sorted(CERTS.glob("*.json")):
        if any(s in p.name for s in (".verify_", ".summary", ".eps")):
            continue
        draw(p, RESULTS / "plots" / f"packing_{p.stem}.png")


if __name__ == "__main__":
    {"headline": headline, "ablation": ablation, "figures": figures, "latex": latex, "cplot": cplot, "syncc": sync_leaderboard_c}[sys.argv[1]]()
