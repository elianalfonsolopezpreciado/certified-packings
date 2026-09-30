# Records reconnaissance (Phase 0)

All values below were looked up on **2026-09-28** (web search / fetch) and are quoted as the
source states them. Nothing here is taken from memory. Where a source gives fewer digits than
we would like, that is stated and the comparison policy accounts for it.

## Problem A - maximise the sum of radii of n disjoint circles in the unit square

**Variables.** `(x_i, y_i, r_i)`, i = 1..n.
**Constraints.** `r_i >= 0`; `r_i <= x_i <= 1 - r_i`; `r_i <= y_i <= 1 - r_i`;
`(x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i + r_j)^2` for i < j.
**Objective.** maximise `sum_i r_i`.

| n | Record (published) | Holder(s) / source | Notes |
|---|--------------------|--------------------|-------|
| 26 | 2.635862 | AlphaEvolve (May 2025); "Best human" column in the TTT-Discover table | first AlphaEvolve value (Friedman's human record was 2.634, per the `RasimAbiyev/circle-packing-explorations` README) |
| 26 | **2.635983** | AlphaEvolve V2; ThetaEvolve (R1-Qwen3-8B); TTT-Discover (Qwen3-8B) - table 3 of arXiv:2601.16175 | 6 digits only |
| 26 | 2.635983099011548 (raw), 2.63597770931127 (under AlphaEvolve's exact checker), 2.6359828390115476 (after shrinking each radius by 1e-8) | ShinkaEvolve, arXiv:2509.19349 | the raw value violates constraints by ~1e-8; the authors themselves report the strictly-checked values |
| 32 | 2.937944 | AlphaEvolve (May 2025) | |
| 32 | **2.939572** | AlphaEvolve V2; TTT-Discover (Qwen3-8B) - table 3 of arXiv:2601.16175 | 6 digits only |

Other claims seen but **not used as the reference** (no primary artifact found):
* A student ("Alex", multi-agent framework "Tactical Maniac v0.5") reportedly reached 2.63592717
  for n=26 in July 2025 (36kr article, 2025-07-21). This is below AlphaEvolve V2 / Shinka
  anyway, and no artifact was available.
* OpenEvolve issue #156 reports 2.6359773947566274 for n=26 (a reproduction, below the V2 value).

### CORRECTION (found later on 2026-09-28): Packomania has an official record table for Problem A

An earlier draft of this file said Packomania has no sum-of-radii category. **That was wrong.**
Packomania maintains "The best known packings of variable-sized circles in a square with maximized
sum of radii" (`csqv`): https://www.packomania.com/csqv/csqv.html , raw table
https://www.packomania.com/csqv/txt/sumradii.txt (12 decimals, retrieved 2026-09-28, stored in
`data/packomania_csqv_sumradii.txt` / `data/packomania_csqv_records.json` (downloaded into the git-ignored `data/` by `scripts/fetch_packomania.py`; NOT redistributed, see `docs/packomania_data.md`); the page says
"complete up to N = 400" and lists submissions through September 2026; per-N page e.g.
https://www.packomania.com/csqv/csqv26.html, dated 03-Sep-2026, author E. Specht). These records
**supersede** the 6-digit AlphaEvolve figures above and exist for every n we study:

| n | Packomania csqv record | note |
|---|---|---|
| 24 | 2.530311586971 | |
| 25 | 2.587275055266 | |
| 26 | **2.635983084919** | AlphaEvolve V2 = "2.635983" (6 digits) is this value truncated |
| 27 | 2.685978684198 | |
| 28 | 2.737739985536 | |
| 29 | 2.790344154631 | |
| 30 | 2.842668747462 | |
| 31 | 2.889969851933 | |
| 32 | **2.939572771205** | AlphaEvolve V2 = "2.939572" |
| 33 | 2.987285008592 | |
| 34 | 3.029799271186 | |
| 35 | 3.074036363730 | |
| 36 | 3.121754486102 | |
| 37 | 3.161498916179 | |
| 38 | 3.205945627974 | |
| 39 | 3.248110798175 | |
| 40 | 3.292391572608 | |

Related: arXiv:2609.05093 ("LLM-Guided Program Evolution for Circle Packing: Breaking 10
Packomania Records", N = 101-114, accepted into Packomania) shows the table is still improved by
new methods, but only for large N; small/medium N have had heavy attention (AlphaEvolve lineage,
FICO Xpress global solver paper arXiv:2605.04850, independent repos such as
`jasonzliang/circle-packing-sota`). We treat n <= 40 records as likely hard/saturated.

**Reference values used by our leaderboard (all n): the Packomania csqv 12-decimal value.**

**Comparison policy (A).** The verifier certifies `sum(r_i - 1e-12)` (documented shrink), so a
certificate of a *tied* configuration is below the record by ~ n*1e-12. Hence for a certified score S:
* **match**: `S >= REC - 1e-11 - n*1e-12` (i.e. raw configuration equals the record to 12 decimals)
* **improvement candidate**: `S > REC + 1e-11` (strict, *after* the shrink) - and only if both
  independent verifier backends pass in fresh processes.
* otherwise **below**.

**Submission channel for A.** Packomania (E. Specht, `eckard.specht@physik.uni-magdeburg.de`,
private communication; the csqv page's update log shows current acceptance). Secondary: AlphaEvolve
problem repository issue/PR. Nothing is ever sent by this project automatically; see `SUBMISSION.md`.

Note on record coordinates: we did **not** download Packomania's coordinate files; all structures
are found by our own search (a "perturbed known structure" init is therefore our own lattice/ring
family and each run's incumbent), which keeps the reproduction independent.

Sources:
* AlphaEvolve V2 / TTT-Discover comparison table: https://arxiv.org/abs/2601.16175 (html: https://arxiv.org/html/2601.16175)
* ShinkaEvolve: https://arxiv.org/abs/2509.19349
* ThetaEvolve: https://arxiv.org/abs/2511.23473
* AlphaEvolve problem repository: https://github.com/google-deepmind/alphaevolve_repository_of_problems
* Georgiev, Gomez-Serrano, Tao, Wagner, "Mathematical exploration and discovery at scale": https://arxiv.org/abs/2511.02864
* Independent SLSQP/basin-hopping baseline (99.94% of human record, no claim): https://github.com/RasimAbiyev/circle-packing-explorations
* OpenEvolve: https://huggingface.co/blog/codelion/openevolve ; issue #156 https://github.com/algorithmicsuperintelligence/openevolve/issues/156
* Out-of-the-box global optimization for packing (FICO Xpress/SCIP): https://arxiv.org/abs/2605.04850

## Problem B - maximise the minimum pairwise distance of n points in the unit square

**Variables.** `p_i = (x_i, y_i) in [0,1]^2`.
**Objective.** maximise `d_n = min_{i<j} ||p_i - p_j||`.
Equivalent to Packomania's "circles in a square" (`csq`): radius `r` of n equal circles in the
unit square relates via `d = 2r / (1 - 2r)`.
(Check: n=10, r=0.148204322565 -> d=0.421279543984, matching the distance table.)

**Record source.** Packomania, http://www.packomania.com/csq/ - column "distance", ASCII table
`http://www.packomania.com/csq/txt/distance.txt` (12 decimals), downloaded 2026-09-28 to
`data/packomania_csq_distance.txt`, parsed into `data/packomania_csq_records.json` (both produced by `scripts/fetch_packomania.py`; not redistributed, see `docs/packomania_data.md`).
Optimality is proven for n <= 30 (Wikipedia, "Circle packing in a square"), so records for
n <= 30 are essentially certain to be optimal and **cannot be beaten**; larger n are conjectured
optimal (heuristic).

Selected values (distance): n=10 0.421279543984, 26 0.238734757241, 32 0.213174562590,
50 0.166526577344, 60 0.149505654049. Full table in the JSON.

**Comparison policy.** Packomania stores 12 decimals; we compare at `1e-11`:
match if `d >= REC - 1e-11`; improvement candidate if `d > REC + 1e-11` and both certificates pass.

**Submission channel.** Packomania's homepage (retrieved 2026-09-28) states:
"Please do not submit any more entries for now; there are over 5,000 new candidates in the queue."
Format per http://www.packomania.com/hints.html: `.pck` file named like `csq<N>.pck`,
line 1 = radius, line 2 = author(s), then one coordinate pair per line with as many decimals as
possible, rescaled to the standard container; contact: Eckard Specht,
`eckard.specht@physik.uni-magdeburg.de`. Submissions are currently closed, so any improvement
would be prepared but **not** sent (see `SUBMISSION.md`).

**Choosing n where the record "looks least optimized".** We have no ground truth about how hard
each record was optimized, so we use an operational proxy: the pilot sweep (Phase 4) measures,
for each n in 31..60, whether our search reproduces the record and how often. n where the record
is reproduced only in rare restarts are treated as hard; n where every method converges to the
record immediately are treated as saturated. This is a heuristic, not evidence.

## Problem C - benchmark from the AlphaEvolve list (only if time remains)

Chosen: first autocorrelation inequality: minimise `max_t (f*f)(t) / (int f)^2` over nonnegative
step functions on [-1/4, 1/4]; any such function gives an upper bound on the constant C1.
For N equal pieces with heights a_i >= 0: `R(a) = 2N * max_m c_m / (sum a)^2`, `c_m = sum_{i+j=m} a_i a_j`.

Published upper bounds (Table 2 of arXiv:2601.16175, retrieved 2026-09-28):

| source | upper bound | pieces |
|---|---|---|
| best human | 1.50973 | 51 |
| AlphaEvolve | 1.50530 | 95 (other papers quote 600) |
| AlphaEvolve V2 | 1.50317 | 1319 |
| ThetaEvolve | 1.50314 | 1319 |
| TTT-Discover | **1.50287** | 30000 |

Reference used by the leaderboard: 1.50287 (tolerance 5e-6; smaller is better). Submission channel: none
(preprint / the AlphaEvolve problem repository). Nothing was submitted. Our results: see README.

## Final targets

1. A: n=26 (REC 2.635983084919), n=32 (REC 2.939572771205); then n in 27..31, 33..40 (addendum);
   every n has a Packomania csqv record (see correction above).
2. B: pilot sweep n=10..60, deep runs on the hardest 3-5 values with n > 30.
3. C: only if budget remains.

---

# Task 2 recon: sum of radii for n >= 100 (retrieved 2026-09-29 09:24 local)

**Reference table.** Packomania csqv `sumradii.txt` (12 decimals), re-downloaded 2026-09-29 09:24
(`data/packomania_csqv_records.json`, not redistributed; the checksums of every snapshot are in `docs/packomania_snapshot_checksums.json`).
It has contiguous entries for n = 1..400 and sparse entries for larger perfect squares (n = 441 ... 10000).
**The table is a fast-moving target:** 59 rows changed between our 2026-09-28 and 2026-09-29 downloads, and the
page's update log (https://www.packomania.com/csqv/csqv.html, read 2026-09-29) lists improvements almost daily by
many groups: E. Specht (D4-symmetric family n = 2k^2-2k+1: 113, 145, 181, ...), Everett Dutton (Gurobi), Jason Liang,
Rasim Abiyev, Byron Tasseff, Wes Sander (n = 101-114 via an LLM-evolved L-BFGS-B optimiser), Yue Huang, Henrik Tallbacka,
Wilfred Heap (huge batches for n = 60..400, latest 29-Sep-2026), Anant Garg, Zeeshan Tariq, Arnold Castro,
Jean-Rene Denoual (n = 54..400, one batch "about 21 hours" of compute). Related preprint: arXiv:2609.05093
(10 records at n = 101..114, LLM cost $27.72; its numbers are already superseded by the table, e.g. n=101 5.289154 -> 5.291233).
Consequence: *final* comparisons are made against a table re-downloaded at the end of the run and the retrieval time is reported.
Any "improvement" must be re-checked against the then-current table (it can be overtaken within hours).

Landmark references (2026-09-29 09:24 download): n=100: 5.263744441843, n=101: 5.291233498851, n=120: 5.776999746200,
n=150: 6.476248191885, n=200: 7.486828249103 (n=200 changed overnight).

**Which n look weak? (heuristic, not evidence).** A robust smooth fit S(n) ~ a*sqrt(n)+b+c/sqrt(n)+d/n over n = 60..400
(excluding perfect squares and the special D4 family; residual std 2.9e-3; the fit was computed inline and its per-n residuals are not stored) ranks the largest
negative residuals (records below the trend): 399, 358, 297, 301, 359, 397, 393, 302, **139**, 392, 303, **138**, 357, **188**, 273, **116**, 396, **215**, **159**...
Those in bold are in the range our tooling can search (n <= ~220). A negative residual can also just mean that n is a structurally
"unlucky" number, so this only guides where to spend the budget.

**Targets chosen.** Landmarks 100, 120, 150, 200 and weak-looking 116, 138, 139, 159, 188 (n <= 200 first). Expectation, stated up front: heavy
competition and much larger compute have already been applied to these n, so a *match* is already a good outcome and an improvement is unlikely.
No n >= 100 was "unreferenced": all n in 100..400 have a Packomania value, so nothing here is labelled a "new benchmark value".

## Task 2 / Task 3 outcome (2026-09-29, last record check 19:02)

* **Sum of radii, n >= 100.** Against the csqv table downloaded at 19:02: **n=120 ahead by +3.03e-4** (ours 5.777303097445 vs 5.776999746200; also ahead at 10:39, 15:46) and
  **n=250 ahead by +1.98e-4** (8.389973532838 vs 8.389775077490; ahead at 15:46, 19:02); n=110 ties (+4.5e-12); the other 17 certified packings are behind
  (`results/large_n_summary.md`). n=150, 188, 200 were ahead of the 09:24 table (+1.4e-4, +6.3e-4, +2.6e-3) but the listed values rose by +1.94e-3, +2.37e-3, +6.99e-3 (listed value at 15:46, identical to the 15:00 reading, minus the 09:24 value), so we are now *behind* the 21:32 table by
  1.80e-3, 1.74e-3, 4.36e-3 and they are not claimed. The table changed in 59 rows between 09-28 and 09-29 09:24 and in 236 further rows by 15:00. Records file checks: Packomania's n=120 configuration (downloaded, not redistributed; a
  different arrangement, listed radii sum 5.776999746196), n=250/150 `sumradii` lines confirmed from their files.
* **Problem C.** Reference 1.50287 (TTT-Discover, 30,000 pieces), 1.50317 (AlphaEvolve V2), 1.50530 (AlphaEvolve), 1.50973 (best human), all from arXiv:2601.16175 Table 2 (retrieved 2026-09-28).
  Our best certified bound: 1.505274644 (N=19,200): better than 1.50530 (by 2.5e-5) and than the best human value, worse than V2 / the 30,000-piece value by 2.0e-3 / 2.4e-3. Not a record.

## Phase 1 hardening outcome (2026-09-29, checks at 21:31-21:32 UTC-6)
* Fresh download (`results/record_check_20260929_213247.txt`): csqv listed n=120: 5.776999746200, n=250: 8.389775077490 (unchanged since 10:39 / 15:46). Ours: 5.7773030974449 / 8.3899735328377 -> AHEAD by +3.034e-4 / +1.985e-4.
* Exact coordinate files downloaded from Packomania (`data/packomania_files/csqv{120,250,27}.txt`, downloaded with `scripts/fetch_packomania.py`, not redistributed); Hungarian assignment under the 8 square symmetries (`python -m search.compare_record <n>`, output in `results/structure_compare_n<n>.txt`, overlays in `results/plots/overlay_n<n>.png`):
  n=27 control: SAME packing (max deviation 1.4e-12); n=120: DIFFERENT (0/120 within 1e-6, 12/120 within 1e-3, max 4.3e-2, median 5.5e-3); n=250: DIFFERENT (0/250 within 1e-6, 129/250 within 1e-3, max 6.3e-2, median 9.2e-4).
* From-scratch third-party validator (`validator/validate_pck.py`, standard library only): `submission/csqv120.pck` and `csqv250.pck` valid with ZERO tolerance, exactly 3N contacts within 3e-12; Packomania's own listed files (converted) fail zero tolerance by ~1e-13
  in squared distance (12-decimal rounding) while having exactly 3N contacts within 3e-12.
* Format: our `.pck` layout matches http://www.packomania.com/hints.html (line 1 largest radius, line 2 author, `x y r` sorted by increasing radius, square side 1 centred at 0). csqv rule: at least 3N contacts (tolerance < 3e-12), met.
  Site state: homepage (2026-09-29 21:31) still says "Please do not submit any more entries for now; there are over 5,000 new candidates in the queue."
* Status: n=120 AHEAD / different structure / validator passed -> claimed. n=250 AHEAD / different structure / validator passed -> claimed. (Outcome (a) of the publication plan.)
