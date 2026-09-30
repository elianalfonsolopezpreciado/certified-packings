# Submission notes

**Nothing has been sent to anyone.** This file lists what could be sent, and where, *if the author decides to*. Only claims that the current certificates, the from-scratch validator
and the structure comparison support are included.

## 0. Before sending anything: re-check the record (it moves hourly)
The Packomania csqv table for n >= 100 changes many times a day (59 rows changed between 09-28 and 09-29 09:24, 236 more by 15:00). Run

```bash
python -m search.record_check
```

immediately before sending. It downloads the live table, appends a time-stamped comparison for every certified packing to `results/record_checks.csv` and prints `AHEAD` only when our exact raw score
exceeds the freshly listed value by more than 1e-11. **Send a claim only if it still prints AHEAD.** (n=27 also prints AHEAD but is a precision artifact, section 3.)

## 1. Status of the two claims (Phase 1 hardening, last check 2026-09-29 21:32 UTC-6, `results/record_check_20260929_213247.txt`)

| case | listed at 21:32 | our exact sum | gain | verdict | structure vs. listed | third-party validator | claimed |
|---|---|---|---|---|---|---|---|
| n=120 | 5.776999746200 | 5.7773030974449 | +3.034e-4 | **AHEAD** | **different** (0/120 circles within 1e-6, 12/120 within 1e-3; max dev 4.3e-2) | passed | yes |
| n=250 | 8.389775077490 | 8.3899735328377 | +1.985e-4 | **AHEAD** | **different** (0/250 within 1e-6, 129/250 within 1e-3; max dev 6.3e-2) | passed | yes |

Evidence: `results/structure_compare_n120.txt`, `results/structure_compare_n250.txt` (Hungarian assignment under the 8 square symmetries, using the exact coordinate files downloaded from Packomania at 21:31),
`results/plots/overlay_n{120,250}.png`; validator: `python validator/validate_pck.py submission/csqv120.pck submission/csqv250.pck --require-contacts` (zero tolerance, exact integers, exactly 3N contacts within 3e-12).
Control: the same comparison on n=27 gives "same packing" (max deviation 1.4e-12), which is why n=27 is *not* claimed.

## 2. Channel: Packomania csqv (E. Specht)
* Contact: `eckard.specht@ovgu.de` (shown on the 2026 csqv pages); the homepage and the hints page (2018) still show `eckard.specht@physik.uni-magdeburg.de`. Check the address before sending.
* **The homepage (2026-09-29 21:31) still says: "Please do not submit any more entries for now; there are over 5,000 new candidates in the queue."** The csqv page's update log shows results arriving by private communication.
  Whether to send now or wait is the author's decision.
* Format (http://www.packomania.com/hints.html, read 2026-09-29): `.pck` file `csqv<N>.pck`; line 1 = largest radius (number only), line 2 = author, then `x y r` per circle sorted by increasing radius, container = square of side 1 centred at (0,0),
  "as many decimal places as possible"; csqv packings must have at least 3N contacts (contact tolerance < 3e-12). Our files: `submission/csqv120.pck`, `submission/csqv250.pck` (20 decimals, radii shrunk by 1e-12, author line "Elian Alfonso Lopez Preciado" - ASCII
  transliteration for compatibility). They were produced with `python scripts/export_pck.py certificates/A_n<N>.json "Elian Alfonso Lopez Preciado" --shrink 1e-12 --digits 20`.
* Email (not sent): the Gmail connector was disconnected when the draft was attempted ("connection invalidated"), so the draft is provided as `submission/email_draft.eml` (opens as an unsent message with both `.pck` files attached; `X-Unsent: 1`) and `submission/email_draft.txt`.

## 3. Not claimed / nothing to submit
| item | status |
|---|---|
| n=27 (csqv) | +2.7e-11 over the listed value but the *same packing* (max deviation 1.4e-12, radii larger by ~1e-12 = the listing's safety margin), see `docs/n27_case.md` |
| n=150, 188, 200 | were ahead of the 09:24 table, overtaken by 15:00 (now -1.8e-3, -1.7e-3, -4.4e-3): not claimed |
| n=110 | tie (+4.5e-12) |
| n = 26..40 (except above), 100..301 others | matched or behind |
| point spreading (csq), n=10..60 | matches only; the csq channel is closed |
| autocorrelation inequality | best certified 1.505274644 (N=19,200): better than 1.50530, above 1.50317 / 1.50287: not a record; nothing to submit |

## 4. Publication packages (all PRIVATE / DRAFT until the author says "publish")
GitHub repository, Hugging Face dataset and Zenodo deposition: see the final report for their status and links. Zenodo DOIs cannot be deleted once published.
