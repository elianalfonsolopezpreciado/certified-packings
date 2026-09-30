# The n = 27 sum-of-radii case (a precision artifact, not a new record)

**What we saw.** Our polished, exactly verified n=27 packing has exact (unshrunk) sum of radii
`2.685978684225346228...`, which is **2.7e-11 above** the Packomania csqv entry `2.685978684198`.
Every other n in 26..40 is within about +-5e-12 of its listed value, so n=27 stands out (27.0 x 1e-12 = 2.7e-11).

**Checks (all reproducible).**
* The exact verifier passes the file with radii shrunk by 1e-30 (both the exact-rational and the 60-digit backend, fresh
  processes): `certificates/sum_radii_n27.eps1e-30.verify_{fraction,mpmath}.json` (certified score 2.685978684225346228...).
  (`--eps 0` fails only at the 1e-41 level because coordinates are printed to 40 digits.)
* Packomania's own n=27 coordinates (download them with `python scripts/fetch_packomania.py --n 27`; the file is not redistributed here, checksum in
  `docs/packomania_snapshot_checksums.json`) recompute to exactly the listed sum 2.685978684198 and are strictly feasible with zero shrink
  (minimum slack 2.8e-13).
* `python -m search.compare_record 27` (Hungarian assignment under the 8 square symmetries): all 27 circles of our packing match Packomania's one-to-one after a 180-degree rotation, with maximum
  coordinate/radius difference **1.4e-12**; our radii are larger by 1.01e-12 on average (0.5-1.4e-12 range).

**Interpretation.** Same packing (same contact structure and positions to ~1e-12). The listed value corresponds to radii that carry a
safety margin of about 1e-12 each, whereas our Newton-polished radii are the exact optimum of the structure. The gain of 2.7e-11 is therefore a
*numerical-precision* effect, not a new arrangement. We do **not** present it as a discovery. If the maintainers want the higher-precision
values for this entry, the certificate is available (see `SUBMISSION.md`); it is their call whether a 2.7e-11 refinement of an identical
structure is worth recording.

**Consequence for the comparison policy.** Our default certificate shrinks every radius by 1e-12 (score - n*1e-12), so with that policy n=27
was reported as a tie (`gap` +3.5e-13) while the unshrunk value shows +2.73e-11 (`gap_raw`). Both columns are in the headline table. No other n
shows this pattern (all other |gap_raw| <= 5.4e-12). For n other than 27 we did not compare coordinates with Packomania's.
