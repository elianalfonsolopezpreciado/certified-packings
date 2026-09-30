# Packomania data: what we used, and how to get it (not redistributed)

This repository does **not** contain any Packomania table or coordinate file. Packomania (E. Specht, https://www.packomania.com) is the property of its maintainer.
Our results only need the *numbers* we cite (reproduced in `results/*.txt`, `results/*.csv`, the paper) plus, for the structure comparison and the record columns,
files that you can download yourself:

```bash
python scripts/fetch_packomania.py            # tables + coordinates of n = 27, 120, 250 into ./data (git-ignored)
python scripts/fetch_packomania.py --n 150    # other coordinate files
make fetch-data                               # same as the first line, failure-tolerant
```

Everything that verifies our claims (`make verify`, `python validator/validate_pck.py ...`, `verify_in_browser/`) works **without** these files. Without them,
`make test` and `make reproduce` still run; the record/gap columns then read `no-published-reference`.

## Files our comparisons used (snapshots)
| file | URL | retrieved | SHA-256 |
|---|---|---|---|
| csq `distance.txt` | http://www.packomania.com/csq/txt/distance.txt | 2026-09-28 | `5e19fa96c45e9969...` |
| csq `radius.txt` | http://www.packomania.com/csq/txt/radius.txt | 2026-09-28 | `621e5749187f6e7a...` |
| csqv `sumradii.txt` | https://www.packomania.com/csqv/txt/sumradii.txt | 2026-09-28 evening | `c356ff3c090ff507...` |
| csqv `sumradii.txt` | same | 2026-09-29 09:24 (UTC-6) | `de1ddcd8a16c74ed...` |
| csqv `sumradii.txt` | same | 2026-09-29 15:46, 19:02 and 21:32 (UTC-6) (identical content) | `4a02dc961c8241ac...` |
| csqv `csqv120.txt` | https://www.packomania.com/csqv/txt/csqv120.txt | 2026-09-29 21:31 (UTC-6) | `7f28b4b4e045450e...` |
| csqv `csqv250.txt` | https://www.packomania.com/csqv/txt/csqv250.txt | 2026-09-29 21:31 (UTC-6) | `7ce80df1766059eb...` |
| csqv `csqv27.txt` | https://www.packomania.com/csqv/txt/csqv27.txt | 2026-09-29 21:31 (UTC-6) | `067f44a3892fc7dc...` |
| hints page | http://www.packomania.com/hints.html | 2026-09-29 21:31 (UTC-6) | `3c7eb576ce43c983...` |

Full 64-hex checksums and byte counts: `docs/packomania_snapshot_checksums.json`. **The tables move** (for csqv n >= 100 several times a day), so a fresh download will usually
not match these hashes; they only identify the snapshots our statements refer to. Every comparison in this project is therefore tied to a download time (see `results/record_check_*.txt`).

## What is derived from Packomania data and kept in the repository
Only our own analysis outputs: the listed *values* for the n we attempted (in `results/large_n_summary.md`, `results/record_checks.csv`, `results/record_check_*.txt`), the deviation statistics
in `results/structure_compare_n*.txt`, and the overlay figures `results/plots/overlay_n*.png` (which draw the listed circles as outlines for comparison). If the maintainer objects to any of these, they can be removed without affecting the certificates.
