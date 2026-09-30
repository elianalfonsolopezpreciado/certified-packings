"""Assemble the (not yet published) release packages from the repository:

    python scripts/build_release.py [--out release]

  release/hf_dataset/   Hugging Face dataset folder: README.md (dataset card with YAML), certificates, .pck files, result tables, validator
  release/zenodo/       Zenodo bundle: paper PDF, code archive (git archive of tag v1.0), certificates archive, .pck files, record-check outputs, .zenodo.json

Nothing is uploaded by this script. Packomania data is NOT included in any package (Hugging Face dataset, Zenodo bundle, code archive): the repository does not contain it;
docs/packomania_data.md lists URLs and checksums and scripts/fetch_packomania.py downloads it on demand.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CARD = """---
license: cc-by-4.0
pretty_name: Certified circle packings (sum of radii) and autocorrelation step functions
language:
- en
tags:
- circle-packing
- extremal-geometry
- mathematics
- certificates
- verification
- packomania
size_categories:
- n<1K
---

# Certified circle packings (sum of radii) and autocorrelation step functions

Author: Elian Alfonso López Preciado, Independent Researcher, León, Guanajuato, México. Code: https://github.com/elianalfonsolopezpreciado/certified-packings (MIT). This dataset (certificates, `.pck` files, tables): CC-BY-4.0.

## What is in here
* `certificates/` - JSON certificates. `A_n<k>.json` / `sum_radii_n<k>.json`: packings of `n` disjoint circles in the unit square [0,1]^2 (`circles: [[x, y, r], ...]`, decimal strings with 40 digits); `min_distance_n<k>.json`: point sets maximising the minimum distance; `C_N<k>.json(.gz)` / `autocorr1_n<k>.json`: step functions (`heights`) for the first autocorrelation inequality. Next to each certificate: the outputs of the independent verifier (`*.verify_fraction.json` exact rational arithmetic, `*.verify_mpmath.json` 60-digit arithmetic).
* `submission/csqv120.pck`, `submission/csqv250.pck` - the two packings in Packomania's `.pck` layout (unit square centred at 0, radii reduced by 1e-12).
* `results/` - `headline_table.md`, `large_n_summary.md`, `record_check_*.txt` (time-stamped comparison with the Packomania table), `structure_compare_n*.txt`.
* `validator/validate_pck.py` - stand-alone validator (standard library only, exact integers, zero tolerance).

## Claims and their limits (please read)
* **Two packings exceed the Packomania "csqv" values that were listed when the table was downloaded on 2026-09-29 at 21:32 (UTC-6):** n=120: 5.7773030974449 (listed 5.776999746200), n=250: 8.3899735328377 (listed 8.389775077490). This is a time-stamped comparison, **not** a "best known" or "world record" statement: the table changes several times a day, and three other packings of ours (n=150, 188, 200) that were ahead in the morning had been overtaken by the afternoon.
* The two packings are structurally different from the listed ones (0/120 and 0/250 circles within 1e-6 after optimal matching under the 8 symmetries of the square), rigid (3n active constraints, positive multipliers) and pass the validator with zero tolerance.
* For the first autocorrelation inequality the best certified bound is 1.505274644 (N=19,200): better than 1.50530 but **above** 1.50317 and 1.50287, so **not a record**.
* Float scores in the search logs can exceed the true value by ~5e-11; only certified values (verifier outputs) count. n=27 looks 2.7e-11 above the listed value but is the same packing with the listing's ~1e-12 safety margin removed and is not claimed.

## Third-party data
No Packomania table or coordinate file is included in this dataset. The comparison values quoted in the result files were read from https://www.packomania.com at the stated times; `docs/packomania_data.md` in the code repository gives the URLs and SHA-256 checksums of the snapshots we used, and `scripts/fetch_packomania.py` downloads them on demand.

## Verify
```bash
python validator/validate_pck.py submission/csqv120.pck submission/csqv250.pck --require-contacts
# certificates: clone the code repository and run `python -m verify certificates/A_n120.json --backend both`
```

## AI-use disclosure
The software implementation, automated testing, experiment execution and a first draft of the documentation were produced with Claude Code using the Claude Sonnet 5.5 model (Anthropic), under the direction of the author. The author defined the research goals, chose the problems and methods to pursue, reviewed the results and the manuscript, and takes full responsibility for the content. The AI system is not an author.

## Citation
See `CITATION.cff` in the code repository; Zenodo DOI: 10.5281/zenodo.23051520 (reserved; it resolves once the deposition is published).
"""

ZEN_README = """Zenodo deposition: 'Two certified sum-of-radii circle packings exceeding the Packomania values listed on 29 September 2026 ...'
Files: main.pdf (preprint), certified-packings-v1.0-code.zip (git archive of tag v1.0: code, tests, certificates, logs, tables),
certificates.zip (certificates + verifier outputs), csqv120.pck / csqv250.pck (Packomania layout), record_check outputs, .zenodo.json (metadata).
Comparisons with Packomania are valid only at the stated download times. Code MIT; data and paper CC-BY-4.0. Packomania tables/coordinate files are NOT included anywhere in this deposition (see docs/packomania_data.md in the code archive: URLs and SHA-256 checksums; scripts/fetch_packomania.py downloads them).
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="release")
    a = ap.parse_args()
    out = ROOT / a.out
    if out.exists():
        shutil.rmtree(out)
    hf, zen = out / "hf_dataset", out / "zenodo"
    for d in (hf / "certificates", hf / "submission", hf / "results", hf / "validator", zen):
        d.mkdir(parents=True, exist_ok=True)
    (hf / "README.md").write_text(CARD, encoding="utf-8")
    for p in (ROOT / "certificates").iterdir():
        if p.is_file() and ".eps" not in p.name:
            shutil.copy2(p, hf / "certificates" / p.name)
    for p in (ROOT / "submission").glob("*.pck"):
        shutil.copy2(p, hf / "submission" / p.name)
    for name in ("headline_table.md", "large_n_summary.md", "record_checks.csv", "C_progress.csv", "ablation.md"):
        if (ROOT / "results" / name).exists():
            shutil.copy2(ROOT / "results" / name, hf / "results" / name)
    for p in list((ROOT / "results").glob("record_check_*.txt")) + list((ROOT / "results").glob("structure_compare_n*.txt")):
        shutil.copy2(p, hf / "results" / p.name)
    shutil.copy2(ROOT / "validator" / "validate_pck.py", hf / "validator" / "validate_pck.py")
    shutil.copy2(ROOT / "LICENSE-DATA-CC-BY-4.0.md", hf / "LICENSE-CC-BY-4.0.md")
    # Zenodo bundle
    shutil.copy2(ROOT / "paper" / "main.pdf", zen / "main.pdf")
    shutil.copy2(ROOT / ".zenodo.json", zen / ".zenodo.json")
    for p in (ROOT / "submission").glob("*.pck"):
        shutil.copy2(p, zen / p.name)
    for p in (ROOT / "results").glob("record_check_*.txt"):
        shutil.copy2(p, zen / p.name)
    subprocess.run(["git", "-c", "safe.directory=" + str(ROOT).replace("\\", "/"), "archive", "--format=zip", "--prefix=certified-packings-v1.0/",
                    "-o", str(zen / "certified-packings-v1.0-code.zip"), "v1.0"], cwd=ROOT, check=True)
    with zipfile.ZipFile(zen / "certificates.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((hf / "certificates").iterdir()):
            z.write(p, "certificates/" + p.name)
    (zen / "README.txt").write_text(ZEN_README, encoding="utf-8")
    print("built", out)
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.parent in (zen,):
            print(f"  {p.relative_to(out)}  {p.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
