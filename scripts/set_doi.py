"""Set the reserved Zenodo DOI (and optionally the publication date) everywhere, then rebuild the derived files.

    python scripts/set_doi.py 10.5281/zenodo.1234567                       # or just the number: 1234567
    python scripts/set_doi.py 10.5281/zenodo.1234567 --date "3 October 2026"

Replaces every `10.5281/zenodo.<id>` (placeholder XXXXXXX or a previous DOI) in: paper/main.tex (macro \\zenododoi), README.md, CITATION.cff, submission/email_draft.txt,
scripts/build_release.py (Hugging Face card). With --date it also sets the \\pubdate macro of the paper. Then it recompiles paper/main.pdf and rebuilds
submission/email_draft.eml. It does NOT commit, tag, push, upload or send anything; the follow-up steps (commit, re-tag v1.0, `python scripts/build_release.py`) are printed at the end.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["paper/main.tex", "README.md", "CITATION.cff", "submission/email_draft.txt", "scripts/build_release.py"]
PAT = re.compile(r"10\.5281/zenodo\.[A-Za-z0-9]+")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("doi", help="e.g. 10.5281/zenodo.1234567 or 1234567")
    ap.add_argument("--date", help="publication date text for the paper, e.g. '3 October 2026'")
    a = ap.parse_args()
    doi = a.doi if a.doi.startswith("10.5281/zenodo.") else "10.5281/zenodo." + a.doi
    if not re.fullmatch(r"10\.5281/zenodo\.[0-9]+", doi):
        print("refusing: the DOI must look like 10.5281/zenodo.<digits>", file=sys.stderr)
        return 2
    total = 0
    for rel in FILES:
        p = ROOT / rel
        s = p.read_text(encoding="utf-8")
        new, k = PAT.subn(doi, s)
        if a.date and rel == "paper/main.tex":
            new, k2 = re.subn(r"(\\newcommand\{\\pubdate\}\{)[^}]*(\})", lambda m: m.group(1) + a.date + m.group(2), new)
            print(f"  {rel}: publication date macro set ({k2} match)")
        if new != s:
            p.write_text(new, encoding="utf-8")
        print(f"  {rel}: {k} DOI occurrence(s) set")
        total += k
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_email_draft.py")], check=True, cwd=ROOT)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=ROOT / "paper", capture_output=True)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=ROOT / "paper", capture_output=True, text=True)
    print("paper recompiled:", "Output written" in r.stdout, "| DOI occurrences replaced:", total)
    print("next (not done by this script): commit, `git tag -d v1.0 && git tag -a v1.0 -m v1.0`, `python scripts/build_release.py`, push to the PRIVATE repo. Nothing was published or sent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
